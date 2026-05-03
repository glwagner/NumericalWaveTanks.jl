using CUDA
using Random
using Statistics
using JLD2
using Oceananigans
using Oceananigans.Units
using Oceananigans.Fields: interpolate
using Printf

# Stratified extension of `extended_les_from_dns.jl` — adds buoyancy
# (BuoyancyTracer) with a tanh thermocline. The DNS state (which has no
# buoyancy field) is still embedded in the upper Lz_dns of the LES
# domain, with the buoyancy initial condition set analytically from the
# tanh profile.
#
# Usage:
#   julia --project ... extended_stratified_les_from_dns.jl <ic_jld2> \
#       snap_idx Nx Ny Nz Lx Ly Lz ϵ α W' stop_time \
#       ρ₁ ρ₂ z_h δ_thermo Q_b
#
# Defaults match L3 of the matrix in claude_code_plan.md, but updated to
# the W23 paper ε = 0.11.

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

function load_dns_snapshot(filepath, idx)
    @info "Loading DNS snapshot $idx from $filepath"
    u_ts = FieldTimeSeries(filepath, "u")
    v_ts = FieldTimeSeries(filepath, "v")
    w_ts = FieldTimeSeries(filepath, "w")
    c_ts = FieldTimeSeries(filepath, "c")
    t_star = u_ts.times[idx]
    @info "  t_star = $t_star s"
    return (; u = u_ts[idx], v = v_ts[idx], w = w_ts[idx], c = c_ts[idx], t_star)
end

function build_extended_stratified_les(arch;
                                       ic_filepath,
                                       snap_idx = 4,
                                       Nx = 192,
                                       Ny = 192,
                                       Nz = 128,
                                       Lx = 0.1,
                                       Ly = 0.1,
                                       Lz = 0.25,
                                       ϵ = 0.11,
                                       k = 2π/0.03,
                                       ν = 1.05e-6,
                                       κ_c = κ_rhodamine,
                                       κ_b = κ_salt,
                                       α = 1.2e-5,
                                       ρ₁ = 1000.0,
                                       ρ₂ = 1035.0,
                                       zₕ = -0.05,
                                       δ_thermo = 0.005,
                                       g = 9.81,
                                       Q_b = 0.0,
                                       stop_time = 60.0,
                                       save_interval = 0.5,
                                       overwrite_existing = true,
                                       name = "ext_strat_les")

    Δb = g * (ρ₂ - ρ₁) / ρ₁
    @info "Stratification: Δb = $Δb m/s², zₕ = $zₕ m, δ = $δ_thermo m"
    @info "Surface buoyancy flux Q_b = $Q_b W/kg"

    dns = load_dns_snapshot(ic_filepath, snap_idx)
    t_star = dns.t_star

    refinement = 1.5
    stretching = 8
    h_(kk) = (kk - 1) / Nz
    ζ₀(kk) = 1 + (h_(kk) - 1) / refinement
    Σ(kk)  = (1 - exp(-stretching * h_(kk))) / (1 - exp(-stretching))

    @info "Building deep LES grid"
    grid = RectilinearGrid(arch,
                           size = (Nx, Ny, Nz),
                           halo = (3, 3, 3),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = kk -> Lz * (ζ₀(kk) * Σ(kk) - 1),
                           topology = (Periodic, Periodic, Bounded))
    @show grid

    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τʷ, parameters = (; α)))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Q_b),
                                    bottom = GradientBoundaryCondition(0))

    a = ϵ / k
    ∂z_uˢ = ConstantStokesShear(a, k)
    u_forcing = Forcing(ν_∂z²_uˢ, parameters = (; ∂z_uˢ, ν))

    @info "Building stratified model"
    model = NonhydrostaticModel(grid;
                                boundary_conditions = (; u = u_bcs, b = b_bcs),
                                advection = WENO(order=5),
                                timestepper = :RungeKutta3,
                                tracers = (:b, :c),
                                buoyancy = BuoyancyTracer(),
                                closure = ScalarDiffusivity(; ν, κ = (b=κ_b, c=κ_c)),
                                stokes_drift = UniformStokesDrift(; ∂z_uˢ),
                                forcing = (; u = u_forcing))

    z_dns_cut = -0.05
    u_dns, v_dns, w_dns, c_dns = dns.u, dns.v, dns.w, dns.c

    function u_ic(x, y, z)
        z < z_dns_cut && return 0.0
        return Float64(interpolate((x, y, z), u_dns))
    end
    function v_ic(x, y, z)
        z < z_dns_cut && return 0.0
        return Float64(interpolate((x, y, z), v_dns))
    end
    function w_ic(x, y, z)
        z < z_dns_cut && return 0.0
        return Float64(interpolate((x, y, z), w_dns))
    end
    function c_ic(x, y, z)
        z < z_dns_cut && return 0.0
        return Float64(interpolate((x, y, z), c_dns))
    end
    # Two-layer buoyancy: light at top (b≈0), dense at bottom (b≈-Δb)
    b_ic(x, y, z) = -Δb/2 * (1 - tanh((z - zₕ) / δ_thermo))

    @info "Setting LES IC: DNS for u/v/w/c (z>$z_dns_cut), tanh for b, zero below"
    set!(model, u=u_ic, v=v_ic, w=w_ic, c=c_ic, b=b_ic)
    model.clock.time = t_star

    simulation = Simulation(model; Δt=1e-4, stop_time)

    max_Δt = 0.05
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

    Nx_, Ny_, Nz_ = size(model.grid)
    file_prefix = @sprintf("%s_t%05d_ep%03d_zh%03d_dr%03d_qb%d_N%d_%d_%d_L%d_%d_%d",
                           name, round(Int, 1000 * t_star),
                           round(Int, 1000 * ϵ),
                           round(Int, 1000 * abs(zₕ)),
                           round(Int, 1000 * (ρ₂ - ρ₁) / ρ₁),
                           round(Int, 1e9 * Q_b),  # 0 for no flux, negative for cooling
                           Nx_, Ny_, Nz_,
                           round(Int, 100 * model.grid.Lx),
                           round(Int, 100 * model.grid.Ly),
                           round(Int, 100 * model.grid.Lz))
    dir = file_prefix
    @info "Saving data to $dir"

    u, v, w = model.velocities
    c = model.tracers.c
    b = model.tracers.b

    U_avg = Field(Average(u, dims=(1, 2)))
    V_avg = Field(Average(v, dims=(1, 2)))
    B_avg = Field(Average(b, dims=(1, 2)))
    C_avg = Field(Average(c, dims=(1, 2)))
    uu = Field(Average(u * u, dims=(1, 2)))
    vv = Field(Average(v * v, dims=(1, 2)))
    ww = Field(Average(w * w, dims=(1, 2)))
    uw = Field(Average(u * w, dims=(1, 2)))
    vw = Field(Average(v * w, dims=(1, 2)))
    wb = Field(Average(w * b, dims=(1, 2)))

    profiles = (U=U_avg, V=V_avg, B=B_avg, C=C_avg,
                uu=uu, vv=vv, ww=ww, uw=uw, vw=vw, wb=wb)
    simulation.output_writers[:profiles] = JLD2Writer(model, profiles; dir, overwrite_existing,
        schedule = TimeInterval(save_interval),
        filename = file_prefix * "_profiles",
        array_type = Array{Float32})

    outputs = merge(model.velocities, model.tracers)
    simulation.output_writers[:xz_left] = JLD2Writer(model, outputs; dir, overwrite_existing,
        schedule = TimeInterval(save_interval),
        filename = file_prefix * "_xz_left",
        indices = (:, 1, :),
        array_type = Array{Float32})

    return simulation
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 1 || error("usage: julia ... <ic_jld2> [snap_idx Nx Ny Nz Lx Ly Lz ϵ α W' stop_time ρ1 ρ2 z_h δ Qb]")
    ic_filepath = ARGS[1]
    snap_idx    = length(ARGS) >= 2  ? parse(Int, ARGS[2])  : 4
    Nx          = length(ARGS) >= 5  ? parse(Int, ARGS[3])  : 192
    Ny          = length(ARGS) >= 5  ? parse(Int, ARGS[4])  : 192
    Nz          = length(ARGS) >= 5  ? parse(Int, ARGS[5])  : 128
    Lx          = length(ARGS) >= 8  ? parse(Float64, ARGS[6])  : 0.1
    Ly          = length(ARGS) >= 8  ? parse(Float64, ARGS[7])  : 0.1
    Lz          = length(ARGS) >= 8  ? parse(Float64, ARGS[8])  : 0.25
    ϵ           = length(ARGS) >= 9  ? parse(Float64, ARGS[9])  : 0.11
    α_arg       = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) * 1e-5 : 1.2e-5
    stop_time   = length(ARGS) >= 12 ? parse(Float64, ARGS[12]) : 60.0
    ρ₁          = length(ARGS) >= 13 ? parse(Float64, ARGS[13]) : 1000.0
    ρ₂          = length(ARGS) >= 14 ? parse(Float64, ARGS[14]) : 1035.0
    zₕ          = length(ARGS) >= 15 ? parse(Float64, ARGS[15]) : -0.05
    δ_thermo    = length(ARGS) >= 16 ? parse(Float64, ARGS[16]) : 0.005
    Q_b         = length(ARGS) >= 17 ? parse(Float64, ARGS[17]) : 0.0

    simulation = build_extended_stratified_les(GPU();
                                               ic_filepath, snap_idx,
                                               Nx, Ny, Nz, Lx, Ly, Lz,
                                               ϵ, α=α_arg,
                                               ρ₁, ρ₂, zₕ, δ_thermo, Q_b,
                                               stop_time)
    run!(simulation)
    @info "Stratified extended LES complete: $simulation"
end
