using CUDA
using Random
using Statistics
using JLD2
using Oceananigans
using Oceananigans.Units
using Oceananigans.Fields: interpolate
using Printf

# Extended LES that picks up from a saved DNS state.
#
# The DNS lives in a 0.2 × 0.2 × 0.1 m domain; this LES domain is
# 0.2 × 0.2 × Lz_les m (default Lz_les = 0.5) — same horizontal footprint,
# deeper. The DNS state is embedded in the upper 0.1 m of the LES domain;
# below that the fluid is at rest. Because the laminar wind-drift boundary
# layer thickness at t≈19 s is only a few mm, the cut at z = -0.1 m is
# essentially continuous (DNS values at z=-0.1 m are already ≈ 0).
#
# Same physical setup as wagner_et_al_2023_les.jl — same wind stress,
# Stokes drift, streaming forcing — but at coarser resolution suitable for
# a long, deep run.
#
# Usage:
#   julia --project experiments/extended_les_from_dns.jl <ic_jld2> <snap_idx> \
#       Nx Ny Nz Lx Ly Lz_les ϵ α W_prime stop_time
#
# Example (uses the DNS produced in the constant_waves run with W'=0.01):
#   julia --project experiments/extended_les_from_dns.jl \
#       constant_waves_ic010000_ep100_k30_alpha120_N768_768_512_L20_20_10/constant_waves_ic010000_ep100_k30_alpha120_N768_768_512_L20_20_10_3d_fields.jld2 \
#       4 192 192 256 0.2 0.2 0.5 0.1 1.2 0.001 60.0
#
# The snap_idx is the 1-indexed snapshot in the DNS 3D fields file. With
# the DNS saved every 1 s from t=16 to t=22, idx=4 → t=19, idx=5 → t=20.

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

function build_extended_les(arch;
                            ic_filepath,
                            snap_idx = 4,           # default: t=19 if DNS saved every 1 s from t=16
                            Nx = 192,
                            Ny = 192,
                            Nz = 256,
                            Lx = 0.1,               # match W23 DNS horizontal extent
                            Ly = 0.1,
                            Lz = 0.5,
                            ϵ = 0.11,               # W23 paper value
                            k = 2π/0.03,
                            ν = 1.05e-6,
                            κ = κ_rhodamine,
                            α = 1.2e-5,
                            stop_time = 60.0,
                            save_interval = 0.5,
                            overwrite_existing = true,
                            name = "ext_les")

    # Load DNS snapshot first — we need t_star for the model clock and the
    # field arrays for the IC interpolation.
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

    a = ϵ / k
    ∂z_uˢ = ConstantStokesShear(a, k)
    u_forcing = Forcing(ν_∂z²_uˢ, parameters = (; ∂z_uˢ, ν))

    @info "Building model (WENO order=5, no subgrid closure)"
    model = NonhydrostaticModel(grid;
                                boundary_conditions = (; u = u_bcs),
                                advection = WENO(order=5),
                                timestepper = :RungeKutta3,
                                tracers = :c,
                                closure = ScalarDiffusivity(; ν, κ),
                                stokes_drift = UniformStokesDrift(; ∂z_uˢ),
                                forcing = (; u = u_forcing))

    # ---- IC: DNS state in the upper 0.1 m, zero below ----
    z_dns_cut = -0.1
    u_dns, v_dns, w_dns, c_dns = dns.u, dns.v, dns.w, dns.c

    # Wrappers that interpolate DNS values when above the cut, return zero below.
    # `interpolate((x, y, z), field)` does trilinear interpolation on the
    # field's own grid (the DNS grid in this case). The Float64 cast is
    # because the DNS file stores Float32 but the LES model uses Float64.
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

    @info "Setting LES IC from DNS snapshot (this takes a minute on CPU interpolation)"
    set!(model, u=u_ic, v=v_ic, w=w_ic, c=c_ic)
    model.clock.time = t_star

    @info "Building simulation"
    simulation = Simulation(model; Δt=1e-4, stop_time)

    Δ = grid.Δxᶜᵃᵃ
    max_Δt = 0.05  # absolute cap (advective CFL is the binding limit)
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

    # ---- Output ----
    Nx_, Ny_, Nz_ = size(model.grid)
    file_prefix = @sprintf("%s_t%05d_ic%06d_ep%03d_alpha%d_N%d_%d_%d_L%d_%d_%d",
                           name, round(Int, 1000 * t_star),
                           round(Int, 1e6 * 0.001),  # placeholder ic id
                           1000ϵ, 1e7 * α,
                           Nx_, Ny_, Nz_,
                           round(Int, 100 * model.grid.Lx),
                           round(Int, 100 * model.grid.Ly),
                           round(Int, 100 * model.grid.Lz))
    dir = file_prefix
    @info "Saving data to $dir"

    u, v, w = model.velocities
    c = model.tracers.c

    # Mean profiles (the primary diagnostic per research_objectives.md)
    U_avg  = Field(Average(u, dims=(1, 2)))
    V_avg  = Field(Average(v, dims=(1, 2)))
    C_avg  = Field(Average(c, dims=(1, 2)))

    # Variance / Reynolds-stress proxies. These are < (u' w') > etc.;
    # because horizontal averages of u, v are NOT subtracted before the
    # product, the saved <u w> equals <u'w'> + U <w> = <u'w'> (since <w>=0).
    # Same for <u u> = U^2 + <u'u'>, so post-process subtracts U^2 to recover
    # the variance.
    uu = Field(Average(u * u, dims=(1, 2)))
    vv = Field(Average(v * v, dims=(1, 2)))
    ww = Field(Average(w * w, dims=(1, 2)))
    uw = Field(Average(u * w, dims=(1, 2)))
    vw = Field(Average(v * w, dims=(1, 2)))

    profiles = (U=U_avg, V=V_avg, C=C_avg,
                uu=uu, vv=vv, ww=ww, uw=uw, vw=vw)

    simulation.output_writers[:profiles] = JLD2Writer(model, profiles; dir, overwrite_existing,
        schedule = TimeInterval(save_interval),
        filename = file_prefix * "_profiles",
        array_type = Array{Float32})

    # One side-view slice for visualization
    outputs = merge(model.velocities, model.tracers)
    simulation.output_writers[:xz_left] = JLD2Writer(model, outputs; dir, overwrite_existing,
        schedule = TimeInterval(save_interval),
        filename = file_prefix * "_xz_left",
        indices = (:, 1, :),
        array_type = Array{Float32})

    return simulation
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        error("usage: julia ... extended_les_from_dns.jl <ic_jld2> [snap_idx Nx Ny Nz Lx Ly Lz ϵ α W' stop_time]")
    end
    ic_filepath = ARGS[1]
    snap_idx    = length(ARGS) >= 2  ? parse(Int, ARGS[2])  : 4
    Nx          = length(ARGS) >= 5  ? parse(Int, ARGS[3])  : 192
    Ny          = length(ARGS) >= 5  ? parse(Int, ARGS[4])  : 192
    Nz          = length(ARGS) >= 5  ? parse(Int, ARGS[5])  : 256
    Lx          = length(ARGS) >= 8  ? parse(Float64, ARGS[6])  : 0.2
    Ly          = length(ARGS) >= 8  ? parse(Float64, ARGS[7])  : 0.2
    Lz          = length(ARGS) >= 8  ? parse(Float64, ARGS[8])  : 0.5
    ϵ           = length(ARGS) >= 9  ? parse(Float64, ARGS[9])  : 0.1
    α_arg       = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) * 1e-5 : 1.2e-5
    # W_prime currently unused in this restart workflow (no random noise added) but kept
    # for CLI parity with the other scripts.
    stop_time   = length(ARGS) >= 12 ? parse(Float64, ARGS[12]) : 60.0

    simulation = build_extended_les(GPU();
                                    ic_filepath, snap_idx,
                                    Nx, Ny, Nz, Lx, Ly, Lz,
                                    ϵ, α=α_arg, stop_time)
    run!(simulation)
    @info "Extended LES complete: $simulation"
end
