using CUDA
using Random
using Statistics
using SpecialFunctions
using OrderedCollections
using JLD2
using Oceananigans
using Oceananigans: fields
using Oceananigans.Units
using Printf

# LES counterpart of experiments/constant_waves.jl (the Wagner et al. 2023 DNS).
# Differences from the DNS:
#   - WENO(order=9) advection acts as the implicit subgrid model (no AMD/Smagorinsky)
#   - molecular ν, κ are kept (no explicit subgrid closure)
#   - initial perturbations are random noise on top of the Ekman-Stokes laminar
#     profile at t₀; no eigenmode loading
# Same domain, same wind stress α√t, same Stokes drift, same streaming forcing.

κ_rhodamine = 1e-7

struct ConstantStokesShear{T}
    a :: T
    k :: T
    ω :: T
end

ConstantStokesShear(a, k; g=9.81, γ=7.2e-5) = ConstantStokesShear{Float64}(a, k, sqrt(g * k + γ * k^3))
@inline (sh::ConstantStokesShear)(z, t) = 2 * sh.a^2 * sh.k^2 * sh.ω * exp(2 * sh.k * z)

@inline ν_∂z²_uˢ(x, y, z, t, p) = - 4 * p.ν * p.∂z_uˢ.a^2 * p.∂z_uˢ.k^3 * p.∂z_uˢ.ω * exp(2 * p.∂z_uˢ.k * z)

@inline τʷ(x, y, t, p) = - p.α * sqrt(t)

function build_les_wave_tank(arch;
                             Nx = 192,
                             Ny = 192,
                             Nz = 128,
                             Lx = 0.2,
                             Ly = 0.2,
                             Lz = 0.1,
                             ϵ = 0.1,
                             k = 2π/0.03,
                             ν = 1.05e-6,
                             κ = κ_rhodamine,
                             α = 1.2e-5,
                             t₀ = 16.0,
                             W′ = 1e-3,
                             stop_time = 22.0,
                             save_interval = 0.2,
                             overwrite_existing = true,
                             name = "w23_les")

    refinement = 1.5
    stretching = 8
    h(k) = (k - 1) / Nz
    ζ₀(k) = 1 + (h(k) - 1) / refinement
    Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

    @info "Building grid"
    grid = RectilinearGrid(arch,
                           size = (Nx, Ny, Nz),
                           halo = (5, 5, 5),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = k -> Lz * (ζ₀(k) * Σ(k) - 1),
                           topology = (Periodic, Periodic, Bounded))

    @show grid

    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τʷ, parameters = (; α)))

    a = ϵ / k
    ∂z_uˢ = ConstantStokesShear(a, k)
    u_forcing = Forcing(ν_∂z²_uˢ, parameters = (; ∂z_uˢ, ν))

    @info "Building model"
    model = NonhydrostaticModel(grid;
                                boundary_conditions = (; u = u_bcs),
                                advection = WENO(order=9),
                                timestepper = :RungeKutta3,
                                tracers = :c,
                                closure = ScalarDiffusivity(; ν, κ),
                                stokes_drift = UniformStokesDrift(; ∂z_uˢ),
                                forcing = (; u = u_forcing))

    ω = ∂z_uˢ.ω
    @info """
        LES wave parameters
            a       = $a
            k       = $k
            2π/k    = $(2π / k)
            ϵ       = $ϵ
            ω       = $ω
    """

    A = α * sqrt(π / 4ν)
    U₀ = A * t₀
    hᵢ = √(2 * ν * t₀)

    function Uᵢ(x, y, z)
        δ = z / hᵢ
        Ξ = W′ * randn()
        return Ξ + U₀ * ((1 + δ^2) * erfc(-δ / √2) + δ * √(2/π) * exp(-δ^2 / 2))
    end

    wᵢ(x, y, z) = W′ * randn()

    Random.seed!(123)
    set!(model, u=Uᵢ, v=wᵢ, w=wᵢ)

    model.clock.time = t₀

    c = model.tracers.c
    view(interior(c), :, :, grid.Nz) .= 1

    @info "Building simulation"
    simulation = Simulation(model; Δt=1e-4, stop_time)

    Δ = grid.Δxᶜᵃᵃ
    max_Δt = 0.1 * Δ^2 / ν
    @show max_Δt
    wizard = TimeStepWizard(; cfl=0.7, max_Δt)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))

    wall_clock = Ref(time_ns())

    function progress(sim)
        umax = maximum(abs, sim.model.velocities.u)
        vmax = maximum(abs, sim.model.velocities.v)
        wmax = maximum(abs, sim.model.velocities.w)
        elapsed = 1e-9 * (time_ns() - wall_clock[])
        @info @sprintf("iter: %d, t: %s, Δt: %s, wall: %s, max|U|=(%.2e, %.2e, %.2e)",
                       iteration(sim), prettytime(time(sim)), prettytime(sim.Δt),
                       prettytime(elapsed), umax, vmax, wmax)
        wall_clock[] = time_ns()
        return nothing
    end
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

    Nx, Ny, Nz = size(model.grid)
    file_prefix = @sprintf("%s_ic%06d_ep%03d_k%d_alpha%d_N%d_%d_%d_L%d_%d_%d",
                           name, 1e6 * W′, 1000ϵ, 1000 * 2π/k, 1e7 * α,
                           Nx, Ny, Nz,
                           100 * model.grid.Lx,
                           100 * model.grid.Ly,
                           100 * model.grid.Lz)

    dir = file_prefix
    @info "Saving data to $dir"

    outputs = merge(model.velocities, model.tracers)
    u, v, w = model.velocities
    U = Field(Average(u, dims=(1, 2)))
    V = Field(Average(v, dims=(1, 2)))
    C = Field(Average(model.tracers.c, dims=(1, 2)))
    uw = Field(Average(u * w, dims=(1, 2)))

    profiles = (u=U, v=V, c=C, uw=uw)

    simulation.output_writers[:profiles] = JLD2Writer(model, profiles; dir, overwrite_existing,
        schedule = TimeInterval(save_interval),
        filename = file_prefix * "_profiles")

    statistics = (u_max = m -> maximum(abs, view(interior(m.velocities.u), :, :, Nz)),
                  v_max = m -> maximum(abs, m.velocities.v),
                  w_max = m -> maximum(abs, m.velocities.w))

    simulation.output_writers[:stats] = JLD2Writer(model, statistics; dir, overwrite_existing,
        schedule = TimeInterval(save_interval),
        filename = file_prefix * "_statistics")

    simulation.output_writers[:yz_left] = JLD2Writer(model, outputs; dir, overwrite_existing,
        schedule = TimeInterval(save_interval),
        filename = file_prefix * "_yz_left",
        indices = (1, :, :))

    simulation.output_writers[:xz_left] = JLD2Writer(model, outputs; dir, overwrite_existing,
        schedule = TimeInterval(save_interval),
        filename = file_prefix * "_xz_left",
        indices = (:, 1, :))

    simulation.output_writers[:xy_top] = JLD2Writer(model, outputs; dir, overwrite_existing,
        schedule = TimeInterval(save_interval),
        filename = file_prefix * "_xy_top",
        indices = (:, :, grid.Nz))

    return simulation
end

if !isinteractive() && abspath(PROGRAM_FILE) == @__FILE__
    parsing = length(ARGS) > 0
    if parsing
        Nx     = parse(Int,     ARGS[1])
        Ny     = parse(Int,     ARGS[2])
        Nz     = parse(Int,     ARGS[3])
        Lx     = parse(Float64, ARGS[4])
        Ly     = parse(Float64, ARGS[5])
        Lz     = parse(Float64, ARGS[6])
        ϵ      = parse(Float64, ARGS[7])
        α_arg  = parse(Float64, ARGS[8]) * 1e-5
        t₀     = parse(Float64, ARGS[9])
        W′     = parse(Float64, ARGS[10])
        stop_time = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) : 22.0

        simulation = build_les_wave_tank(GPU();
                                         Nx, Ny, Nz, Lx, Ly, Lz,
                                         α=α_arg, ϵ, t₀, W′, stop_time)
    else
        simulation = build_les_wave_tank(GPU())
    end

    run!(simulation)

    @info "LES complete: $simulation"
    for (n, w) in simulation.output_writers
        @info "OutputWriter $n: $(abspath(w.filepath))"
    end
end
