using CUDA
using Random
using Statistics
using SpecialFunctions
using OrderedCollections
using JLD2
using MPI
using Oceananigans
using Oceananigans: fields
using Oceananigans.Forcings: regularize_forcing
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.DistributedComputations: Distributed, Partition, mpi_rank
using Oceananigans.Units
using Printf

# Stratified DNS counterpart of constant_waves.jl. Same Wagner et al. 2023
# DNS setup (Centered(2), molecular ν, κ; wave-averaged Craik-Leibovich
# equations with prescribed monochromatic Stokes drift), plus a
# BuoyancyTracer with a tanh thermocline initial condition. Use this when
# you want a stratified DNS to use as IC for stratified LES.
#
# CLI:
#   julia ... stratified_dns.jl Nx Ny Nz Lx Ly Lz ϵ α t₀ W' [stop_time] \
#       [ρ₁ ρ₂ zₕ δ_thermo Q_b]
#
# Example (W23 paper params + thermocline at z=-3 cm):
#   julia --project experiments/stratified_dns.jl \
#       768 768 512 0.1 0.1 0.05 0.11 1.2 16.0 0.005 22.0 1000 1035 -0.03 0.005 0.0

function select_architecture(Px, Py, Pz)
    if Px * Py * Pz > 1
        MPI.Initialized() || MPI.Init()
        return Distributed(GPU(); partition = Partition(Px, Py, Pz))
    else
        return GPU()
    end
end

is_root() = !MPI.Initialized() || MPI.Comm_rank(MPI.COMM_WORLD) == 0

κ_rhodamine = 1e-7
κ_salt      = 1e-7

struct ConstantStokesShear{T}
    a :: T
    k :: T
    ω :: T
end

ConstantStokesShear(a, k; g=9.81, γ=7.2e-5) = ConstantStokesShear{Float64}(a, k, sqrt(g * k + γ * k^3))
@inline (sh::ConstantStokesShear)(z, t) = 2 * sh.a^2 * sh.k^2 * sh.ω * exp(2 * sh.k * z)

@inline ν_∂z²_uˢ(x, y, z, t, p) = - 4 * p.ν * p.∂z_uˢ.a^2 * p.∂z_uˢ.k^3 * p.∂z_uˢ.ω * exp(2 * p.∂z_uˢ.k * z)

@inline τʷ(x, y, t, p) = - p.α * sqrt(t)

function build_stratified_dns(arch;
                              Ny = 768,
                              Ly = 0.1,
                              Nx = Ny,
                              Lx = Ly,
                              Nz = 512,
                              Lz = 0.05,
                              ϵ = 0.11,
                              k = 2π/0.03,
                              ν = 1.05e-6,
                              κ_c = κ_rhodamine,
                              κ_b = κ_salt,
                              α = 1.2e-5,
                              t₀ = 16.0,
                              W′ = 0.005,
                              ρ₁ = 1000.0,
                              ρ₂ = 1035.0,
                              zₕ = -0.03,
                              δ_thermo = 0.005,
                              g = 9.81,
                              Q_b = 0.0,
                              stop_time = 22.0,
                              save_interval = 0.5,
                              save_interval_3d = 1.0,
                              overwrite_existing = true,
                              name = "strat_dns")

    Δb = g * (ρ₂ - ρ₁) / ρ₁
    is_root() && @info "Stratification: Δb = $Δb m/s², zₕ = $zₕ m, δ_thermo = $δ_thermo m, Q_b = $Q_b"

    refinement = 1.5
    stretching = 8
    h(kk) = (kk - 1) / Nz
    ζ₀(kk) = 1 + (h(kk) - 1) / refinement
    Σ(kk)  = (1 - exp(-stretching * h(kk))) / (1 - exp(-stretching))

    is_root() && @info "Building grid"
    grid = RectilinearGrid(arch,
                           size = (Nx, Ny, Nz),
                           halo = (3, 3, 3),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = kk -> Lz * (ζ₀(kk) * Σ(kk) - 1),
                           topology = (Periodic, Periodic, Bounded))
    is_root() && @show grid

    # Boundary conditions
    u_top_bc = FluxBoundaryCondition(τʷ, parameters = (; α))
    u_bcs = FieldBoundaryConditions(top = u_top_bc)
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Q_b),
                                    bottom = GradientBoundaryCondition(0))

    # Stokes drift + streaming forcing
    a = ϵ / k
    ∂z_uˢ = ConstantStokesShear(a, k)
    u_forcing = Forcing(ν_∂z²_uˢ, parameters = (; ∂z_uˢ, ν))

    vitd = VerticallyImplicitTimeDiscretization()

    is_root() && @info "Building model"
    model = NonhydrostaticModel(grid;
                                boundary_conditions = (; u = u_bcs, b = b_bcs),
                                advection = Centered(order=2),
                                timestepper = :RungeKutta3,
                                tracers = (:b, :c),
                                buoyancy = BuoyancyTracer(),
                                closure = ScalarDiffusivity(vitd; ν, κ = (b=κ_b, c=κ_c)),
                                stokes_drift = UniformStokesDrift(; ∂z_uˢ),
                                forcing = (; u = u_forcing))

    ω = ∂z_uˢ.ω
    is_root() && @info """

        Wave parameters | Values
        =============== | ======
                      a | $a
                      k | $k
                 2π / k | $(2π / k)
                      ϵ | $ϵ
                      ω | $ω
    """

    # Initial condition: laminar Ekman-Stokes profile + noise + tanh buoyancy
    A = α * sqrt(π / 4ν)
    U₀ = A * t₀
    hᵢ = √(2 * ν * t₀)

    function Uᵢ(x, y, z)
        δ = z / hᵢ
        Ξ = 10W′ * randn()
        return Ξ + U₀ * ((1 + δ^2) * erfc(-δ / √2) + δ * √(2/π) * exp(-δ^2 / 2))
    end
    wᵢ(x, y, z) = 10W′ * randn()
    bᵢ(x, y, z) = -Δb/2 * (1 - tanh((z - zₕ) / δ_thermo))

    set!(model, u=Uᵢ, v=wᵢ, w=wᵢ, b=bᵢ)
    model.clock.time = t₀

    c = model.tracers.c
    view(interior(c), :, :, grid.Nz) .= 1

    is_root() && @info "Revvving up a simulation..."
    simulation = Simulation(model; Δt=1e-4, stop_time)

    Δ = grid.Δxᶜᵃᵃ
    max_Δt = 0.1 * Δ^2 / ν
    is_root() && @show max_Δt
    wizard = TimeStepWizard(; cfl=1.0, max_Δt)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))

    wall_clock = Ref(time_ns())
    function progress(sim)
        umax = maximum(abs, sim.model.velocities.u)
        vmax = maximum(abs, sim.model.velocities.v)
        wmax = maximum(abs, sim.model.velocities.w)
        bmin = minimum(sim.model.tracers.b)
        bmax = maximum(sim.model.tracers.b)
        elapsed = 1e-9 * (time_ns() - wall_clock[])
        is_root() && @info @sprintf("iter: %d, t: %s, Δt: %s, wall: %s, max|U|=(%.2e, %.2e, %.2e), b∈[%.3e, %.3e]",
                       iteration(sim), prettytime(time(sim)), prettytime(sim.Δt),
                       prettytime(elapsed), umax, vmax, wmax, bmin, bmax)
        wall_clock[] = time_ns()
        return nothing
    end
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

    # Output (slim — same template as the slimmed constant_waves.jl)
    Nx_, Ny_, Nz_ = size(model.grid)
    file_prefix = @sprintf("%s_ic%06d_ep%03d_zh%03d_dr%03d_qb%d_alpha%d_N%d_%d_%d_L%d_%d_%d",
                           name, round(Int, 1e6 * W′), round(Int, 1000 * ϵ),
                           round(Int, 1000 * abs(zₕ)),
                           round(Int, 1000 * (ρ₂ - ρ₁) / ρ₁),
                           round(Int, 1e9 * Q_b),
                           round(Int, 1e7 * α),
                           Nx_, Ny_, Nz_,
                           round(Int, 100 * model.grid.Lx),
                           round(Int, 100 * model.grid.Ly),
                           round(Int, 100 * model.grid.Lz))

    rank_suffix = MPI.Initialized() ? @sprintf("_rank%03d", MPI.Comm_rank(MPI.COMM_WORLD)) : ""
    dir = file_prefix
    is_root() && @info "Saving data to $dir"

    outputs = merge(model.velocities, model.tracers)
    u, v, w = model.velocities
    C = Field(Average(model.tracers.c, dims=(1, 2)))
    U = Field(Average(u, dims=(1, 2)))
    B = Field(Average(model.tracers.b, dims=(1, 2)))

    statistics = (u_max = m -> maximum(abs, view(interior(m.velocities.u), :, :, Nz_)),
                  v_max = m -> maximum(abs, m.velocities.v),
                  w_max = m -> maximum(abs, m.velocities.w),
                  b_min = m -> minimum(m.tracers.b),
                  b_max = m -> maximum(m.tracers.b))

    simulation.output_writers[:avg] = JLD2Writer(model, (c=C, u=U, b=B); dir, overwrite_existing,
        schedule = TimeInterval(save_interval),
        filename = file_prefix * "_averages" * rank_suffix,
        array_type = Array{Float32})

    simulation.output_writers[:stats] = JLD2Writer(model, statistics; dir, overwrite_existing,
        schedule = TimeInterval(save_interval),
        filename = file_prefix * "_statistics" * rank_suffix,
        array_type = Array{Float32})

    simulation.output_writers[:xz_left] = JLD2Writer(model, outputs; dir, overwrite_existing,
        schedule = TimeInterval(save_interval),
        filename = file_prefix * "_xz_left" * rank_suffix,
        indices = (:, 1, :),
        array_type = Array{Float32})

    simulation.output_writers[:fields_3d] = JLD2Writer(model, outputs; dir, overwrite_existing,
        schedule = TimeInterval(save_interval_3d),
        filename = file_prefix * "_3d_fields" * rank_suffix,
        array_type = Array{Float32},
        with_halos = false)

    return simulation
end

if abspath(PROGRAM_FILE) == @__FILE__
    Nx        = parse(Int,     ARGS[1])
    Ny        = parse(Int,     ARGS[2])
    Nz        = parse(Int,     ARGS[3])
    Lx        = parse(Float64, ARGS[4])
    Ly        = parse(Float64, ARGS[5])
    Lz        = parse(Float64, ARGS[6])
    ϵ         = parse(Float64, ARGS[7])
    α_arg     = parse(Float64, ARGS[8]) * 1e-5
    t₀        = parse(Float64, ARGS[9])
    W′        = parse(Float64, ARGS[10])
    stop_time = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) : 22.0
    ρ₁        = length(ARGS) >= 12 ? parse(Float64, ARGS[12]) : 1000.0
    ρ₂        = length(ARGS) >= 13 ? parse(Float64, ARGS[13]) : 1035.0
    zₕ        = length(ARGS) >= 14 ? parse(Float64, ARGS[14]) : -0.03
    δ_thermo  = length(ARGS) >= 15 ? parse(Float64, ARGS[15]) : 0.005
    Q_b       = length(ARGS) >= 16 ? parse(Float64, ARGS[16]) : 0.0
    Px        = length(ARGS) >= 19 ? parse(Int,     ARGS[17]) : 1
    Py        = length(ARGS) >= 19 ? parse(Int,     ARGS[18]) : 1
    Pz        = length(ARGS) >= 19 ? parse(Int,     ARGS[19]) : 1

    arch = select_architecture(Px, Py, Pz)
    is_root() && @info "Architecture: $arch"

    simulation = build_stratified_dns(arch;
                                      Nx, Ny, Nz, Lx, Ly, Lz,
                                      ϵ, α=α_arg, t₀, W′,
                                      ρ₁, ρ₂, zₕ, δ_thermo, Q_b,
                                      stop_time)
    run!(simulation)

    if is_root()
        @info "Stratified DNS complete: $simulation"
        for (n, w) in simulation.output_writers
            @info "OutputWriter $n: $(abspath(w.filepath))"
        end
    end
end
