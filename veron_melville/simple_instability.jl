# TODO:
#
# 1. Parameter study in near-laboratory conditions (increasing wind...)
# 2. Tune parameters (ϵ, β, k) to match transition time to turbulence in the lab
# 3. Weak forcing limit / long-time interactions between waves and turbulence
# 4. When does Craik-Leibovich break? When does it not work? Do we need lots and lots of wave breaking?

using CUDA
using GLMakie
using Random
using Statistics
using OrderedCollections
using JLD2
using Oceananigans
using Oceananigans: fields
using Oceananigans.Forcings: regularize_forcing
using Oceananigans.Units
using Printf

@info "Building a grid..." 

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

model = NonhydrostaticModel(arch,
                            advection = WENO5(),
                            timestepper = :RungeKutta3,
                            grid = grid,
                            tracers = :c,
                            boundary_conditions = boundary_conditions,
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

function run_numerical_wave_tank!(model; ϵ=0.0, k=2π/0.03, max_velocity=Dict())
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
                                        
    simulation = Simulation(model, Δt=1e-4, stop_time=60seconds)

    wizard = TimeStepWizard(cfl=0.5, max_Δt=1.0)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

    max_velocity[(ϵ, k)] = []

    function progress(sim)

        umax = maximum(abs, sim.model.velocities.u)
        vmax = maximum(abs, sim.model.velocities.v)
        wmax = maximum(abs, sim.model.velocities.w)

        t = time(sim)
        h = √(ν * t)
        Re = umax * h / ν

        push!(max_velocity[(ϵ, k)], (t=t, U=(umax, vmax, wmax), Re=Re))

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

    prefix = @sprintf("%s_%d_%d_%d_k%.1e_ep%.1e", name, grid.Nx, grid.Ny, grid.Nz, k, ϵ)

    outputs = merge(model.velocities, model.tracers)

    Nz = grid.Nz

    statistics = (u_max = model -> maximum(abs, view(interior(model.velocities.u), :, :, Nz)), 
                  u_min = model -> minimum(abs, view(interior(model.velocities.u), :, :, Nz)),
                  v_max = model -> maximum(abs, view(interior(model.velocities.v), :, :, Nz)),
                  w_max = model -> maximum(abs, model.velocities.w))

    save_interval = 0.02

    simulation.output_writers[:yz_left] = JLD2OutputWriter(model, outputs,
                                                           schedule = TimeInterval(save_interval),
                                                           force = true,
                                                           prefix = prefix * "_yz_left",
                                                           field_slicer = FieldSlicer(i = 1))

    simulation.output_writers[:xz_left] = JLD2OutputWriter(model, outputs,
                                                           schedule = TimeInterval(save_interval),
                                                           force = true,
                                                           prefix = prefix * "_xz_left",
                                                           field_slicer = FieldSlicer(j = 1))

    simulation.output_writers[:xy_bottom] = JLD2OutputWriter(model, outputs,
                                                             schedule = TimeInterval(save_interval),
                                                             force = true,
                                                             prefix = prefix * "_xy_bottom",
                                                             field_slicer = FieldSlicer(k = 1))

    simulation.output_writers[:yz_right] = JLD2OutputWriter(model, outputs,
                                                            schedule = TimeInterval(save_interval),
                                                            force = true,
                                                            prefix = prefix * "_yz_right",
                                                            field_slicer = FieldSlicer(i = grid.Nx))

    simulation.output_writers[:xz_right] = JLD2OutputWriter(model, outputs,
                                                            schedule = TimeInterval(save_interval),
                                                            force = true,
                                                            prefix = prefix * "_xz_right",
                                                            field_slicer = FieldSlicer(j = grid.Ny))

    simulation.output_writers[:xy_top] = JLD2OutputWriter(model, outputs,
                                                          schedule = TimeInterval(save_interval),
                                                          force = true,
                                                          prefix = prefix * "_xy_top",
                                                          field_slicer = FieldSlicer(k = grid.Nz))

    simulation.output_writers[:averages] = JLD2OutputWriter(model, (c=C, u=U, η²=E²),
                                                            schedule = TimeInterval(save_interval),
                                                            force = true,
                                                            prefix = prefix * "_averages")

    simulation.output_writers[:statistics] = JLD2OutputWriter(model, statistics,
                                                              schedule = TimeInterval(save_interval),
                                                              force = true,
                                                              prefix = prefix * "_statistics")

    run!(simulation)

    @info "Simulation complete: $simulation. Output:"

    for (name, writer) in simulation.output_writers
        absfilepath = abspath(writer.filepath)
        @info "OutputWriter $name, $absfilepath:\n $writer"
    end

    return nothing
end

max_velocity = Dict()
epsilons = [3e-1, 2e-1, 1e-1]
wavenumbers = [2π / 0.02, 2π / 0.03]

#####
##### Run experiments
#####

run_numerical_wave_tank!(model; max_velocity) # reference

for ϵ in epsilons, k in wavenumbers
    run_numerical_wave_tank!(model; ϵ, k, max_velocity)
end

#####
##### Analyze data
#####

#=
Δz = Array(grid.Δzᵃᵃᶠ[1:Nz+1])

function analyze!(analysis, ϵ, k)
    #prefix = @sprintf("%s_%d_%d_%d_k%.1e_ep%.1e", name, grid.Nx, grid.Ny, grid.Nz, k, ϵ)
    prefix = @sprintf("%s_%d_%d_%d_k%.1e_ep%.1e", "increasing_wind_instability", grid.Nx, grid.Ny, grid.Nz, k, ϵ)
    yz_filepath = prefix * "_yz.jld2"
    statistics_filepath = prefix * "_statistics.jld2"
    averages_filepath = prefix * "_averages.jld2"

    yz_file = jldopen(yz_filepath)
    statistics_file = jldopen(statistics_filepath)
    averages_file = jldopen(averages_filepath)

    iterations = parse.(Int, keys(averages_file["timeseries/t"]))
    
    η² = Float64[]
    max_w = Float64[]
    max_u = Float64[]
    uz₀ = Float64[]
    t = Float64[]

    for iter in iterations
        push!(η², sum(averages_file["timeseries/η²/$iter"][1, 1, :] .* Δz))
        push!(t, averages_file["timeseries/t/$iter"])

        U = averages_file["timeseries/u/$iter"][1, 1, :]
        surface_uz = (U[Nz] - U[Nz-1]) / Δz[Nz]
        push!(uz₀, surface_uz)

        push!(max_u, statistics_file["timeseries/u_max/$iter"][1, 1, 1])
        push!(max_w, statistics_file["timeseries/w_max/$iter"][1, 1, 1])
    end

    analysis[(ϵ, k)] = (; t, η², max_w, max_u, uz₀)

    return nothing
end

analysis = OrderedDict()

ν = 1.05e-6
ϵ = 0.0
k = 2π / 0.02
analyze!(analysis, ϵ, k)

for ϵ in epsilons
    for k in wavenumbers
        analyze!(analysis, ϵ, k)
    end
end

fig = Figure(resolution=(1600, 800))
ax = Axis(fig[1, 1:4],# yscale=log10,
          xlabel="Time (s)", ylabel="maximum(w) (m s⁻¹)")

ϵ★s  = Float64[]
t★s  = Float64[]
k★s  = Float64[]
h★s  = Float64[]
ϖ★s  = Float64[]
Re★s = Float64[]
La★s = Float64[]
Sh★s = Float64[]

for (ϵ, k) in keys(analysis)
    reference = analysis[(0.0, 2π/0.02)]
    data = analysis[(ϵ, k)]
    big_max_w = map((ref_w, w) -> w / 1.5 > ref_w, reference.max_w, data.max_w)
    i★ = findfirst(big_max_w)

    if !(isnothing(i★)) && ϵ > 0.01

        t★ = data.t[i★]
        h★ = √(ν * t★)
        u★ = data.max_u[i★]
        Re★ = u★ * h★ / ν

        uz₀★ = data.uz₀[i★]
        a = ϵ / k
        sh = ConstantStokesShear(a, k)
        uˢz = 2 * sh.a^2 * sh.k^2 * sh.ω

        ϖ★ = ϵ^2 * sh.ω * h★ / u★

        push!(Re★s, Re★)
        push!(ϵ★s, ϵ)
        push!(k★s, k)
        push!(h★s, h★)
        push!(t★s, t★)
        push!(Sh★s, uˢz / uz₀★)
        push!(La★s, ν / h★^2 / (ϵ^2 * sh.ω))
        push!(ϖ★s, ϖ★)
        max_w★ = data.max_w[i★]
        scatter!(ax, [t★], [max_w★])
    end

    t = data.t
    #max_w = data.max_w
    max_u = data.max_u
    λ = 2π / k
    #lines!(ax, t, max_w, label="ϵ = $ϵ, λ = $λ")
    lines!(ax, t, max_u, label="ϵ = $ϵ, λ = $λ")
end

axislegend(ax, position=:rt)

ax = Axis(fig[2, 1], xlabel="1 / Re = ν / u h", ylabel="ϵ") #, xscale=log10, yscale=log10)
scatter!(ax, 1 ./ Re★s, ϵ★s, color=k★s)

ax = Axis(fig[2, 2], xlabel="1 / Re = ν / u h", ylabel="1 / k h")#, yscale=log10, xscale=log10)
scatter!(ax, 1 ./ Re★s, 1 ./ (k★s .* h★s), color=k★s)

ax = Axis(fig[2, 3], xlabel="ν / u h (k h)^(1/8)", ylabel="ϵ")#, yscale=log10, xscale=log10)
scatter!(ax, 1 ./ ((k★s .* h★s).^(1/8) .* Re★s), ϵ★s, color=k★s)

ax = Axis(fig[2, 4], xlabel="1 / Re = ν / u h", ylabel="ϵ² ω h / u = ∂ᶻuˢ / ∂ᶻu (shear ratio)", yscale=log10, xscale=log10)
scatter!(ax, 1 ./ Re★s, ϖ★s, color=k★s)

display(fig)
=#

#####
##### Pretty movie
#####

#=
fig_w = Figure(resolution=(1600, 800))

yv = CUDA.@allowscalar Array(ynodes(Face, grid))
zw = CUDA.@allowscalar Array(znodes(Face, grid))
yw = yu = CUDA.@allowscalar Array(ynodes(Center, grid))
zu = zv = CUDA.@allowscalar Array(znodes(Center, grid))

n = Node(1)

for (i, ϵ) in enumerate(epsilons[1:4])
    k = wavenumbers[1]
    prefix = @sprintf("%s_%d_%d_%d_k%.1e_ep%.1e", name, grid.Nx, grid.Ny, grid.Nz, k, ϵ)
    filepath = prefix * "_yz.jld2"
    file = jldopen(filepath)

    iterations = parse.(Int, keys(file["timeseries/t"])) 
    @show length(iterations)

    u = @lift file["timeseries/u/" * string(iterations[$n])][1, :, :]
    v = @lift file["timeseries/v/" * string(iterations[$n])][1, :, :]
    w = @lift file["timeseries/w/" * string(iterations[$n])][1, :, :]
    t(n) = file["timeseries/t/" * string(iterations[n])]

    title_str = @lift "ϵ = $ϵ, t = $(prettytime(t($n)))"

    ax_u = Axis(fig_w[1, i], title=title_str)
    heatmap!(ax_u, yu, zu, u, colormap=:balance)

    ax_v = Axis(fig_w[2, i], title=title_str)
    heatmap!(ax_v, yv, zv, v, colormap=:balance)

    ax_w = Axis(fig_w[3, i], title=title_str)
    heatmap!(ax_w, yw, zw, w, colormap=:balance)
end

#display(fig_w)

nframes = 3000 #length(iterations)
GLMakie.record(fig_w, "increasing_wind.mp4", 1:nframes, framerate=8) do ni
    @info "Plotting frame $ni of $nframes..."
    n[] = ni
end
=#
