using CUDA
using Random
using Statistics
using OrderedCollections
using JLD2
using Oceananigans
using Oceananigans: fields
using Oceananigans.Forcings: regularize_forcing
using Oceananigans.Units
using Printf

κ_rhodamine = 1e-9 # find a reference for this

# Constant Stokes shear utility
struct ConstantStokesShear{T}
    a :: T
    k :: T
    ω :: T
end

ConstantStokesShear(a, k; g=9.81, T=7.2e-5) = ConstantStokesShear{Float64}(a, k, sqrt(g * k + T * k^3))
@inline (sh::ConstantStokesShear)(z, t) = 2 * sh.a^2 * sh.k^2 * sh.ω * exp(2 * sh.k * z) 

# Stokes streaming
@inline ν_∂z²_uˢ(x, y, z, t, p) = - 4 * p.ν * p.∂z_uˢ.a^2 * p.∂z_uˢ.k^3 * p.∂z_uˢ.ω * exp(2 * p.∂z_uˢ.k * z)

function build_numerical_wave_tank(arch;
                                   # Cross-stream direction
                                   Ny = 256,
                                   Ly = 0.2,
                                   # Along-stream direction
                                   Nx = Ny,
                                   Lx = Ly,
                                   # Vertical direction
                                   Nz = round(Int, Ny/2),
                                   Lz = Ly / 2,
                                   ϵ = 0.0,
                                   k = 2π/0.03,
                                   ν = 1.05e-6,
                                   κ = κ_rhodamine,
                                   β = 1.1e-5,
                                   stop_time = 30.0,
                                   save_interval = 0.2,
                                   overwrite_existing = false,
                                   name = "increasing_wind")

    refinement = 1.5 # controls spacing near surface (higher means finer spaced)
    stretching = 8   # controls rate of stretching at bottom
    h(k) = (k - 1) / Nz
    ζ₀(k) = 1 + (h(k) - 1) / refinement
    Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

    @info "Building a grid..." 
    grid = RectilinearGrid(arch,
                           size = (Nx, Ny, Nz),
                           halo = (3, 3, 3),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = k -> Lz * (ζ₀(k) * Σ(k) - 1), # (-Lz, 0)
                           topology = (Periodic, Periodic, Bounded))

    @show grid

    #####
    ##### Surface stress
    #####

    @inline τʷ(x, y, t) = - β * sqrt(t)
    u_top_bc = FluxBoundaryCondition(τʷ)
        
    u_bcs = FieldBoundaryConditions(top = u_top_bc)
    boundary_conditions = (; u = u_bcs)

    #####
    ##### Stokes streaming term with a forcing function
    #####
    
    a = ϵ / k
    ∂z_uˢ = ConstantStokesShear(a, k)
    u_forcing = Forcing(ν_∂z²_uˢ, parameters=(; ∂z_uˢ, ν))

    #####
    ##### The model
    #####

    model = NonhydrostaticModel(; grid, boundary_conditions,
                                advection = WENO(),
                                timestepper = :RungeKutta3,
                                tracers = :c,
                                closure = ScalarDiffusivity(; ν, κ),
                                stokes_drift = UniformStokesDrift(; ∂z_uˢ),
                                forcing = (; u = u_forcing))

    ω = ∂z_uˢ.ω

    @info """

        Wave parameters | Values
        =============== | ======
                      a | $a
                      k | $k
                 2π / k | $(2π / k)
                      ϵ | $ϵ
    """

    model.clock.time = 0
    model.clock.iteration = 0

    Random.seed!(123)
    uᵢ(x, y, z) = 1e-11 * randn()
    set!(model, u=uᵢ, v=uᵢ, w=uᵢ)

    c = model.tracers.c
    view(interior(c), :, :, grid.Nz) .= 1

    @info "Revvving up a simulation..."
                                        
    simulation = Simulation(model; Δt=1e-4, stop_time)

    Δ = min(minimum(parent(grid.Δzᵃᵃᶜ)), grid.Δxᶜᵃᵃ)
    @show max_Δt = 0.1 * Δ^2 / ν
    wizard = TimeStepWizard(; cfl=0.3, max_Δt)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))

    wall_clock = Ref(time_ns())

    # Closure over wall_clock
    function progress(sim)

        umax = maximum(abs, sim.model.velocities.u)
        vmax = maximum(abs, sim.model.velocities.v)
        wmax = maximum(abs, sim.model.velocities.w)

        t = time(sim)
        h = √(ν * t)
        Re = umax * h / ν
        elapsed = 1e-9 * (time_ns() - wall_clock[])

        @info @sprintf("Time: %s, iter: %d, Δt: %s, wall time: %s, max|U|: (%.2e, %.2e, %.2e)  m s⁻¹, Re: %.1f",
                       prettytime(t),
                       iteration(sim),
                       prettytime(sim.Δt),
                       prettytime(elapsed),
                       umax, vmax, wmax,
                       Re)

        wall_clock[] = time_ns()

        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    #####
    ##### Set up output
    #####

    Nx, Ny, Nz = size(model.grid)

    file_prefix = @sprintf("%s_ep%d_k%d_beta%d_N%d_%d_%d_L%d_%d_%d",
                           name, 1000ϵ, 1000 * 2π/k, 1e7 * β,
                           Nx, Ny, Nz,
                           100 * model.grid.Lx,
                           100 * model.grid.Ly,
                           100 * model.grid.Lz)

    nobackup_dir = "."
    dir = joinpath(nobackup_dir, file_prefix)

    @info "Saving data to $file_prefix"

    outputs = merge(model.velocities, model.tracers)

    u, v, w = model.velocities
    η² = (∂z(v) - ∂y(w))^2
    C = Field(Average(model.tracers.c, dims=(1, 2)))
    U = Field(Average(u, dims=(1, 2)))
    E² = Field(Average(η², dims=(1, 2)))

    simulation.output_writers[:avg] = JLD2OutputWriter(model, (c=C, u=U, η²=E²); dir, overwrite_existing,
                                                       schedule = TimeInterval(save_interval),
                                                       filename = file_prefix * "_averages")

    simulation.output_writers[:fast_avg] = JLD2OutputWriter(model, (c=C, u=U, η²=E²); dir, overwrite_existing,
                                                            schedule = TimeInterval(0.02),
                                                            filename = file_prefix * "_hi_freq_averages")

    Nz = grid.Nz

    statistics = (u_max = model -> maximum(abs, view(interior(model.velocities.u), :, :, Nz)), 
                  u_min = model -> minimum(abs, view(interior(model.velocities.u), :, :, Nz)),
                  v_max = model -> maximum(abs, view(interior(model.velocities.v), :, :, Nz)),
                  w_max = model -> maximum(abs, model.velocities.w))

    simulation.output_writers[:stats] = JLD2OutputWriter(model, statistics; dir, overwrite_existing,
                                                         schedule = TimeInterval(save_interval),
                                                         filename = file_prefix * "_statistics")

    simulation.output_writers[:hi_freq_stats] = JLD2OutputWriter(model, statistics; dir, overwrite_existing,
                                                                 schedule = TimeInterval(0.02),
                                                                 filename = file_prefix * "_hi_freq_statistics")

    simulation.output_writers[:yz_left] = JLD2OutputWriter(model, outputs; dir, overwrite_existing,
                                                           schedule = TimeInterval(save_interval),
                                                           filename = file_prefix * "_yz_left",
                                                           indices = (1, :, :))

    simulation.output_writers[:xz_left] = JLD2OutputWriter(model, outputs; dir, overwrite_existing,
                                                           schedule = TimeInterval(save_interval),
                                                           filename = file_prefix * "_xz_left",
                                                           indices = (:, 1, :))

    simulation.output_writers[:xy_bottom] = JLD2OutputWriter(model, outputs; dir, overwrite_existing,
                                                             schedule = TimeInterval(save_interval),
                                                             filename = file_prefix * "_xy_bottom",
                                                             indices = (:, :, 1))

    simulation.output_writers[:yz_right] = JLD2OutputWriter(model, outputs; dir, overwrite_existing,
                                                            schedule = TimeInterval(save_interval),
                                                            filename = file_prefix * "_yz_right",
                                                            indices = (grid.Nx, :, :))

    simulation.output_writers[:xz_right] = JLD2OutputWriter(model, outputs; dir, overwrite_existing,
                                                            schedule = TimeInterval(save_interval),
                                                            filename = file_prefix * "_xz_right",
                                                            indices = (:, grid.Ny, :))

    simulation.output_writers[:xy_top] = JLD2OutputWriter(model, outputs; dir, overwrite_existing,
                                                          schedule = TimeInterval(save_interval),
                                                          filename = file_prefix * "_xy_top",
                                                          indices = (:, :, grid.Nz))

    simulation.output_writers[:chk] = Checkpointer(model; dir, overwrite_existing,
                                                   schedule = TimeInterval(0.5),
                                                   cleanup = true,
                                                   prefix = file_prefix * "_checkpointer")

    return simulation
end

parsing = true

# For example:
# julia --project constant_waves.jl 384 384 256 0.1 0.1 0.05 0.13 1.2 false
#                                   Nx  Ny  Nz  Lx  Ly  Lz   ϵ    β   pickup
# julia --project constant_waves.jl 32 32 32 0.1 0.1 0.05 0.13 1.2 false

if parsing
    Nx     = parse(Int,     ARGS[1])
    Ny     = parse(Int,     ARGS[2])
    Nz     = parse(Int,     ARGS[3])
    Lx     = parse(Float64, ARGS[4])
    Ly     = parse(Float64, ARGS[5])
    Lz     = parse(Float64, ARGS[6])
    ϵ      = parse(Float64, ARGS[7])
    β      = parse(Float64, ARGS[8]) * 1e-5
    pickup = parse(Bool,    ARGS[9])
end

@show overwrite_existing = !pickup
arch = GPU()

simulation = build_numerical_wave_tank(arch; Nx, Ny, Nz, Lx, Ly, Lz, β, overwrite_existing, ϵ, k=2π/0.03)

run!(simulation; pickup)

@info "Simulation complete: $simulation. Output:"

for (name, writer) in simulation.output_writers
    if !(writer isa Checkpointer)
        absfilepath = abspath(writer.filepath)
        @info "OutputWriter $name, $absfilepath:\n $writer"
    end
end

