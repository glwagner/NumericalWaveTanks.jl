using CUDA
using Random
using Statistics
using SpecialFunctions
using OrderedCollections
using JLD2
using Oceananigans
using Oceananigans: fields
using Oceananigans.Forcings: regularize_forcing
using Oceananigans.Units
using Printf

@inline function mean_velocity(x, y, z, t, p)
    h = √(2 * p.ν * p.t₀)
    U₀ = p.A * p.t₀
    δ = z / h
    return U₀ * ((1 + δ^2) * erfc(-δ / √2) + δ * √(2/π) * exp(-δ^2 / 2))
end

function decaying_turbulence_on_shear_flow(arch;
                                           # Cross-stream direction
                                           Ny = 256,
                                           Ly = 0.2,
                                           # Along-stream direction
                                           Nx = Ny,
                                           Lx = Ly,
                                           # Vertical direction
                                           Nz = round(Int, Ny/2),
                                           Lz = Ly / 2,
                                           ν = 1.05e-6,
                                           β = 1.2e-5,
                                           initial_time = 0.0,
                                           stop_time = 1.0,
                                           save_interval = stop_time / 10,
                                           overwrite_existing = true,
                                           name = "weakish_initial_turbulence")

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

    @show A = β * sqrt(π / 4ν)
    t₀ = initial_time
    U = BackgroundField(mean_velocity, parameters=(; A, ν, t₀))

    model = NonhydrostaticModel(; grid,
                                background_fields = (; u=U),
                                advection = CenteredSecondOrder(),
                                timestepper = :RungeKutta3,
                                closure = ScalarDiffusivity(; ν))

    # Set initial condition
    Random.seed!(123)
    wᵢ(x, y, z) = 2.0 * A * t₀ * randn()
    set!(model, u=wᵢ, v=wᵢ, w=wᵢ)

    @info "Revvving up a simulation..."
    simulation = Simulation(model; Δt=1e-5, stop_time)

    Δ = min(minimum(parent(grid.Δzᵃᵃᶜ)), grid.Δxᶜᵃᵃ)
    @show max_Δt = 0.1 * Δ^2 / ν
    wizard = TimeStepWizard(; cfl=0.5, max_Δt)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(3))
    wall_clock = Ref(time_ns())

    # Closure over wall_clock
    function progress(sim)
        umax = maximum(abs, sim.model.velocities.u)
        vmax = maximum(abs, sim.model.velocities.v)
        wmax = maximum(abs, sim.model.velocities.w)

        t = time(sim)
        elapsed = 1e-9 * (time_ns() - wall_clock[])

        @info @sprintf("Time: %s, iter: %d, Δt: %s, wall time: %s, max|U|: (%.2e, %.2e, %.2e)  m s⁻¹",
                       prettytime(t),
                       iteration(sim),
                       prettytime(sim.Δt),
                       prettytime(elapsed),
                       umax, vmax, wmax)

        wall_clock[] = time_ns()

        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    #####
    ##### Set up output
    #####

    Nx, Ny, Nz = size(model.grid)

    file_prefix = @sprintf("%s_beta%d_t0%d_N%d_%d_%d_L%d_%d_%d",
                           name, 1e7 * β, initial_time,
                           Nx, Ny, Nz,
                           100 * model.grid.Lx,
                           100 * model.grid.Ly,
                           100 * model.grid.Lz)

    nobackup_dir = "."
    dir = joinpath(nobackup_dir, file_prefix)

    @info "Saving data to $file_prefix"
    schedule = SpecifiedTimes(0.1, 0.2, 0.5, 1.0)
    simulation.output_writers[:fields] = JLD2OutputWriter(model, model.velocities; dir,
                                                          overwrite_existing, schedule,
                                                          with_halos = true,
                                                          filename = file_prefix * "_fields")

    statistics = (u_max = model -> maximum(abs, view(interior(model.velocities.u), :, :, Nz)), 
                  u_min = model -> minimum(abs, view(interior(model.velocities.u), :, :, Nz)),
                  v_max = model -> maximum(abs, view(interior(model.velocities.v), :, :, Nz)),
                  w_max = model -> maximum(abs, model.velocities.w))

    simulation.output_writers[:hi_freq_stats] = JLD2OutputWriter(model, statistics; dir, overwrite_existing,
                                                                 schedule = TimeInterval(1e-3),
                                                                 filename = file_prefix * "_hi_freq_statistics")

    simulation.output_writers[:yz_left] = JLD2OutputWriter(model, model.velocities; dir, overwrite_existing,
                                                           schedule = TimeInterval(save_interval),
                                                           filename = file_prefix * "_yz_left",
                                                           indices = (1, :, :))

    simulation.output_writers[:xz_left] = JLD2OutputWriter(model, model.velocities; dir, overwrite_existing,
                                                           schedule = TimeInterval(save_interval),
                                                           filename = file_prefix * "_xz_left",
                                                           indices = (:, 1, :))

    simulation.output_writers[:xy_top] = JLD2OutputWriter(model, model.velocities; dir, overwrite_existing,
                                                          schedule = TimeInterval(save_interval),
                                                          filename = file_prefix * "_xy_top",
                                                          indices = (:, :, grid.Nz))


    return simulation
end

# For example:
# julia --project initial_turbulence.jl 768 768 512 0.2 0.2 0.1 1.2 16
# julia --project initial_turbulence.jl 384 384 256 0.1 0.1 0.05 1.2 16
# julia --project initial_turbulence.jl 768 768 512 0.1 0.1 0.05 1.2 16
#                                       Nx  Ny  Nz  Lx   Ly  Lz   β t₀

#=
parsing = false
Nx = Ny = 64
Nz = 48
Lx = Ly = 0.2
Lz = 0.1
β = 1.2e-5
t₀ = 16

arch = CPU()
=#

parsing = true
arch = GPU()

if parsing
    Nx     = parse(Int,     ARGS[1])
    Ny     = parse(Int,     ARGS[2])
    Nz     = parse(Int,     ARGS[3])
    Lx     = parse(Float64, ARGS[4])
    Ly     = parse(Float64, ARGS[5])
    Lz     = parse(Float64, ARGS[6])
    β      = parse(Float64, ARGS[7]) * 1e-5
    t₀     = parse(Float64, ARGS[8])
end

simulation = decaying_turbulence_on_shear_flow(arch; Nx, Ny, Nz, Lx, Ly, Lz, β, initial_time = t₀)
run!(simulation)

@info "Simulation complete: $simulation. Output:"

for (name, writer) in simulation.output_writers
    if !(writer isa Checkpointer)
        absfilepath = abspath(writer.filepath)
        @info "OutputWriter $name, $absfilepath:\n $writer"
    end
end

