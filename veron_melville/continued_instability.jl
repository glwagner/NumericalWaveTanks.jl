using CUDA
#using GLMakie
using Random
using Statistics
using OrderedCollections
using JLD2
using Oceananigans
using Oceananigans.Architectures: architecture, arch_array
using Oceananigans: fields
using Oceananigans.Forcings: regularize_forcing
using Oceananigans.Units
using Printf

Ny = 256
Ly = 0.2

Nx = Ny
Lx = Ly

Nz = Ny   # number of points in the vertical direction
Lz = Ly/2 # domain depth

refinement = 1.5 # controls spacing near surface (higher means finer spaced)
stretching = 8   # controls rate of stretching at bottom
h(k) = (k - 1) / Nz
ζ₀(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

arch = GPU()

grid = RectilinearGrid(arch,
                       size = (Nx, Ny, Nz),
                       halo = (3, 3, 3),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = k -> Lz * (ζ₀(k) * Σ(k) - 1), # (-Lz, 0)
                       topology = (Periodic, Periodic, Bounded))

@show grid

#####
##### Parameters
#####

#name = "constant_wind"
name = "increasing_wind"

κ = 1e-6 # tracer diffusivity
const ν = 1.05e-6    # m² s⁻¹, kinematic viscosity

struct ConstantStokesShear{T}
    a :: T
    k :: T
    ω :: T
end

ConstantStokesShear(a, k; g = 9.81, T = 7.2e-5) =
    ConstantStokesShear{Float64}(a, k, sqrt(g * k + T * k^3))

@inline (sh::ConstantStokesShear)(z, t) = 2 * sh.a^2 * sh.k^2 * sh.ω * exp(2 * sh.k * z) 

surface_stokes_drift(sh) = sh.a^2 * sh.k * sh.ω

#####
##### Surface stress
#####

u_top_bc = name == "increasing_wind" ?
    FluxBoundaryCondition((x, y, t) -> - 1e-5 * sqrt(t)) :
    FluxBoundaryCondition(-1e-4)

u_bcs = FieldBoundaryConditions(top = u_top_bc)
boundary_conditions = (; u = u_bcs)

#####
##### Stokes streaming
#####

@inline ν_∂z²_uˢ(x, y, z, t, sh) = - 4 * ν * sh.a^2 * sh.k^3 * sh.ω * exp(2 * sh.k * z)
u_forcing = Forcing(ν_∂z²_uˢ, parameters=ConstantStokesShear(0.0, 0.0))

model = NonhydrostaticModel(; grid, boundary_conditions,
                            advection = WENO5(),
                            timestepper = :RungeKutta3,
                            tracers = :c,
                            closure = IsotropicDiffusivity(ν=ν, κ=κ),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=ConstantStokesShear(0.0, 0.0)),
                            forcing = (; u = u_forcing),
                            coriolis = nothing,
                            buoyancy = nothing)

u, v, w = model.velocities
η² = (∂z(v) - ∂y(w))^2

C = Field(Average(model.tracers.c, dims=(1, 2)))
U = Field(Average(model.velocities.u, dims=(1, 2)))
E² = Field(Average(η², dims=(1, 2)))

function build_numerical_wave_tank(model; ϵ=0.0, k=2π/0.03, stop_time=60.0, prefix="")
    a = ϵ / k
    ∂z_uˢ = ConstantStokesShear(a, k)

    u_forcing = Forcing(ν_∂z²_uˢ, parameters=∂z_uˢ)
    u = model.velocities.u
    field_names = keys(fields(model))
    u_forcing = regularize_forcing(u_forcing, u, :u, field_names)

    forcing_names = keys(model.forcing)
    forcings = Dict(name => model.forcing[name] for name in forcing_names)
    forcings[:u] = u_forcing 

    # Replace forcing and Stokes drift
    model.forcing = NamedTuple(name => forcings[name] for name in forcing_names)
    model.stokes_drift = UniformStokesDrift(; ∂z_uˢ)

    ν = model.closure.ν
    ω = ∂z_uˢ.ω

    @info """

        Wave parameters | Values
        =============== | ======
                      a | $a
                      k | $k
                 2π / k | $(2π / k)
                      ϵ | $ϵ
    """
    #                 La | $La
    #               La⁻¹ | $(1 / La)

    model.clock.time = 0
    model.clock.iteration = 0

    Random.seed!(123)
    uᵢ(x, y, z) = 1e-11 * randn()
    set!(model, u=uᵢ, v=uᵢ, w=uᵢ)

    c = model.tracers.c
    view(interior(c), :, :, grid.Nz) .= 1

    @info "Revvving up a simulation..."
                                        
    simulation = Simulation(model; Δt=1e-4, stop_time)

    wizard = TimeStepWizard(cfl=0.5, max_Δt=1.0)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

    function progress(sim)

        umax = maximum(abs, sim.model.velocities.u)
        vmax = maximum(abs, sim.model.velocities.v)
        wmax = maximum(abs, sim.model.velocities.w)

        t = time(sim)
        h = √(ν * t)
        Re = umax * h / ν

        @info @sprintf("Time: %s, iteration: %d, next Δt: %s, max|U|: (%.2e, %.2e, %.2e)  m s⁻¹, hk: %.2f, Re: %.1f",
                       prettytime(t),
                       iteration(sim),
                       prettytime(sim.Δt),
                       umax, vmax, wmax,
                       h * k,
                       Re)

        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    #####
    ##### Set up output
    #####

    file_prefix = @sprintf("%s%s_%d_%d_%d_k%.1e_ep%.1e", prefix, name, grid.Nx, grid.Ny, grid.Nz, k, ϵ)

    outputs = merge(model.velocities, model.tracers)

    Nz = grid.Nz

    statistics = (u_max = model -> maximum(abs, view(interior(model.velocities.u), :, :, Nz)), 
                  u_min = model -> minimum(abs, view(interior(model.velocities.u), :, :, Nz)),
                  v_max = model -> maximum(abs, view(interior(model.velocities.v), :, :, Nz)),
                  w_max = model -> maximum(abs, model.velocities.w))

    save_interval = 0.02

    simulation.output_writers[:yz] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(save_interval),
                                                      force = true,
                                                      prefix = file_prefix * "_yz",
                                                      field_slicer = FieldSlicer(i = 1))

    simulation.output_writers[:xz] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(save_interval),
                                                      force = true,
                                                      prefix = file_prefix * "_xz",
                                                      field_slicer = FieldSlicer(j = 1))

    simulation.output_writers[:xy] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(save_interval),
                                                      force = true,
                                                      prefix = file_prefix * "_xy",
                                                      field_slicer = FieldSlicer(k = grid.Nz))

    simulation.output_writers[:averages] = JLD2OutputWriter(model, (c=C, u=U, η²=E²),
                                                            schedule = TimeInterval(save_interval),
                                                            force = true,
                                                            prefix = file_prefix * "_averages")

    simulation.output_writers[:statistics] = JLD2OutputWriter(model, statistics,
                                                              schedule = TimeInterval(save_interval),
                                                              force = true,
                                                              prefix = file_prefix * "_statistics")

    #=
    simulation.output_writers[:fields] = JLD2OutputWriter(model, fields(model),
                                                          schedule = SpecifiedTimes(20:30...),
                                                          force = true,
                                                          field_slicer = nothing,
                                                          prefix = file_prefix * "_fields")
    =#


    return simulation
end

filename = "transition_spin_upincreasing_wind_256_256_256_k2.1e+02_ep1.0e-01_fields.jld2"
start_iter = 7828 # t ≈ 27

function run_numerical_wave_tank!(model; ϵ=0.0, k=2π/0.03, stop_time=60.0, prefix="continued_")
    simulation = build_numerical_wave_tank(model; ϵ, k, stop_time, prefix)

    arch = architecture(model.grid)

    file = jldopen(filename)
    uᵢ = arch_array(arch, file["timeseries/u/$start_iter"])
    vᵢ = arch_array(arch, file["timeseries/v/$start_iter"])
    wᵢ = arch_array(arch, file["timeseries/w/$start_iter"])
    cᵢ = arch_array(arch, file["timeseries/c/$start_iter"])
    tᵢ = file["timeseries/t/$start_iter"]
    close(file)

    @show tᵢ
    @show size(uᵢ)
    @show size(vᵢ)
    @show size(wᵢ)
    @show size(cᵢ)
    @show size(model.velocities.u)
    @show size(model.velocities.v)
    @show size(model.velocities.w)
    @show size(model.tracers.c)

    model.velocities.u .= uᵢ
    model.velocities.v .= vᵢ
    model.velocities.w .= wᵢ
    model.tracers.c .= cᵢ
    model.clock.time = tᵢ 

    run!(simulation)

    @info "Simulation complete: $simulation. Output:"

    for (name, writer) in simulation.output_writers
        absfilepath = abspath(writer.filepath)
        @info "OutputWriter $name, $absfilepath:\n $writer"
    end

    return nothing
end

run_numerical_wave_tank!(model, ϵ=0.0, k=2π/0.03, stop_time=60, prefix="continued_")
run_numerical_wave_tank!(model, ϵ=3e-1, k=2π/0.03, stop_time=60, prefix="continued_")
run_numerical_wave_tank!(model, ϵ=1e-1, k=2π/0.03, stop_time=60, prefix="continued_")

