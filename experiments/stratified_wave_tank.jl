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

# Stratified extension of experiments/wagner_et_al_2023_les.jl.
# Adds a passive buoyancy tracer with a two-layer (tanh) initial profile.
# Same forcing, same domain, same WENO(9) implicit-LES numerics.
#
# Density convention: b = -g (ρ - ρ₀) / ρ₀, with ρ₀ = ρ_upper.
# So upper layer has b ≈ 0, lower (denser) layer has b ≈ -Δb.

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

function build_stratified_wave_tank(arch;
                                    Nx = 192,
                                    Ny = 192,
                                    Nz = 128,
                                    Lx = 0.2,
                                    Ly = 0.2,
                                    Lz = 0.1,
                                    ϵ = 0.1,
                                    k = 2π/0.03,
                                    ν = 1.05e-6,
                                    κ_c = κ_rhodamine,
                                    κ_b = κ_salt,
                                    α = 1.2e-5,
                                    t₀ = 16.0,
                                    W′ = 1e-3,
                                    ρ₁ = 1000.0,
                                    ρ₂ = 1035.0,
                                    zₕ = -0.03,
                                    δ_thermo = 0.005,
                                    g = 9.81,
                                    stop_time = 22.0,
                                    save_interval = 0.2,
                                    overwrite_existing = true,
                                    name = "stratified")

    Δb = g * (ρ₂ - ρ₁) / ρ₁

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
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0),
                                    bottom = GradientBoundaryCondition(0))

    a = ϵ / k
    ∂z_uˢ = ConstantStokesShear(a, k)
    u_forcing = Forcing(ν_∂z²_uˢ, parameters = (; ∂z_uˢ, ν))

    @info "Building model"
    model = NonhydrostaticModel(grid;
                                boundary_conditions = (; u = u_bcs, b = b_bcs),
                                advection = WENO(order=9),
                                timestepper = :RungeKutta3,
                                tracers = (:b, :c),
                                buoyancy = BuoyancyTracer(),
                                closure = ScalarDiffusivity(; ν, κ = (b=κ_b, c=κ_c)),
                                stokes_drift = UniformStokesDrift(; ∂z_uˢ),
                                forcing = (; u = u_forcing))

    @info """
        Stratified LES wave parameters
            ρ₁, ρ₂  = $ρ₁, $ρ₂  kg/m³
            Δb      = $Δb  m/s²
            zₕ      = $zₕ  m
            δ_thermo= $δ_thermo  m
            ϵ       = $ϵ
            α       = $α
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
    bᵢ(x, y, z) = -Δb/2 * (1 - tanh((z - zₕ) / δ_thermo))

    Random.seed!(123)
    set!(model, u=Uᵢ, v=wᵢ, w=wᵢ, b=bᵢ)

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
        bmin = minimum(sim.model.tracers.b)
        bmax = maximum(sim.model.tracers.b)
        elapsed = 1e-9 * (time_ns() - wall_clock[])
        @info @sprintf("iter: %d, t: %s, Δt: %s, wall: %s, max|U|=(%.2e, %.2e, %.2e), b∈[%.3e, %.3e]",
                       iteration(sim), prettytime(time(sim)), prettytime(sim.Δt),
                       prettytime(elapsed), umax, vmax, wmax, bmin, bmax)
        wall_clock[] = time_ns()
        return nothing
    end
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

    Nx, Ny, Nz = size(model.grid)
    file_prefix = @sprintf("%s_dr%03d_zh%03d_ep%03d_alpha%d_N%d_%d_%d_L%d_%d_%d",
                           name, round(Int, 1000 * (ρ₂ - ρ₁) / ρ₁),
                           round(Int, 1000 * abs(zₕ)),
                           1000ϵ, 1e7 * α,
                           Nx, Ny, Nz,
                           100 * model.grid.Lx,
                           100 * model.grid.Ly,
                           100 * model.grid.Lz)

    dir = file_prefix
    @info "Saving data to $dir"

    outputs = merge(model.velocities, model.tracers)
    u, v, w = model.velocities
    b = model.tracers.b
    U  = Field(Average(u, dims=(1, 2)))
    V  = Field(Average(v, dims=(1, 2)))
    B  = Field(Average(b, dims=(1, 2)))
    C  = Field(Average(model.tracers.c, dims=(1, 2)))
    uw = Field(Average(u * w, dims=(1, 2)))
    wb = Field(Average(w * b, dims=(1, 2)))

    profiles = (u=U, v=V, b=B, c=C, uw=uw, wb=wb)

    simulation.output_writers[:profiles] = JLD2Writer(model, profiles; dir, overwrite_existing,
        schedule = TimeInterval(save_interval),
        filename = file_prefix * "_profiles")

    statistics = (u_max = m -> maximum(abs, view(interior(m.velocities.u), :, :, Nz)),
                  v_max = m -> maximum(abs, m.velocities.v),
                  w_max = m -> maximum(abs, m.velocities.w),
                  b_min = m -> minimum(m.tracers.b),
                  b_max = m -> maximum(m.tracers.b))

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
        ρ₁     = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) : 1000.0
        ρ₂     = length(ARGS) >= 12 ? parse(Float64, ARGS[12]) : 1035.0
        zₕ     = length(ARGS) >= 13 ? parse(Float64, ARGS[13]) : -0.03
        stop_time = length(ARGS) >= 14 ? parse(Float64, ARGS[14]) : 22.0

        simulation = build_stratified_wave_tank(GPU();
                                                Nx, Ny, Nz, Lx, Ly, Lz,
                                                α=α_arg, ϵ, t₀, W′,
                                                ρ₁, ρ₂, zₕ, stop_time)
    else
        simulation = build_stratified_wave_tank(GPU())
    end

    run!(simulation)

    @info "Stratified LES complete: $simulation"
    for (n, w) in simulation.output_writers
        @info "OutputWriter $n: $(abspath(w.filepath))"
    end
end
