#####
##### Makie-free utilities for the regular-wave cases (Experiments 2 and 3): loading the
##### horizontally averaged profiles, the null-corrected Eulerian change
#####   ΔU(z, t) = (U_waves+turb − U_turb) − (U_waves+null − U_null),
##### the FOV window, ensembles over seeds, and the turbulence calibration report.
#####
##### Command line: julia --project=. analysis/anti_stokes/regular_common.jl report=<turbulence_control dir>
#####

include("common.jl")

struct RegularRun
    dir  :: String
    meta :: Dict{String, Any}
    fts  :: Dict{String, Any}
end

function load_regular_run(dir; fields=("U", "V", "W", "UU", "VV", "WW", "UW"))
    isdir(dir) || error("Run directory $dir does not exist")
    meta = load(joinpath(dir, "metadata.jld2"))
    path = joinpath(dir, "profiles.jld2")
    fts = Dict{String, Any}(name => FieldTimeSeries(path, name) for name in fields)
    return RegularRun(dir, meta, fts)
end

times(run::RegularRun) = collect(Float64, run.fts[first(keys(run.fts))].times)
run_grid(run::RegularRun) = run.fts[first(keys(run.fts))].grid
znodes_centers(run::RegularRun) = collect(Float64, Array(znodes(run_grid(run), Center())))
znodes_faces(run::RegularRun) = collect(Float64, Array(znodes(run_grid(run), Face())))
Δz_centers(run::RegularRun) = diff(znodes_faces(run))
run_case(run::RegularRun) = run.meta["case"]
t_fov(run::RegularRun) = Float64(run.meta["t_FOV"])
k₀(run::RegularRun) = Float64(run_case(run).k)

"Horizontally averaged profile `name` as an `(Nz, Nt)` array (Nz + 1 for z-face fields)."
zt(run::RegularRun, name) = Array{Float64}(Array(interior(run.fts[name]))[1, 1, :, :])

"The prescribed Stokes drift on the cell centres."
stokes_profile(run::RegularRun) = [Float64(run.meta["waves"].Uˢ₀) * exp(2Float64(run.meta["waves"].k) * zk) for zk in znodes_centers(run)]

"FOV window: t_FOV ± half-width (default 2.5 s)."
fov_window(run::RegularRun; half=2.5) = (t_fov(run) - half, t_fov(run) + half)

function window_mean(A::AbstractMatrix, t, t₀, t₁)
    idx = findall(τ -> t₀ - 1e-6 <= τ <= t₁ + 1e-6, t)
    isempty(idx) && error("No output times in [$t₀, $t₁]")
    return vec(mean(A[:, idx]; dims=2))
end

"Eulerian variances and covariance (z-profiles in time) from the horizontally averaged moments."
function regular_moments(run::RegularRun)
    U, V, W = zt(run, "U"), zt(run, "V"), zt(run, "W")
    uu = zt(run, "UU") .- U.^2
    vv = zt(run, "VV") .- V.^2
    ww = zt(run, "WW") .- W.^2
    Wc = 0.5 .* (W[1:end-1, :] .+ W[2:end, :])
    uw = zt(run, "UW") .- U .* Wc
    return (; uu, vv, ww, uw)
end

"""
    regular_residual(waves_dir, control_dir, null_dir=nothing, quiescent_dir=nothing)

ΔU(z, t) and the waves and control runs.
"""
function regular_residual(waves_dir, control_dir, null_dir=nothing, quiescent_dir=nothing)
    wv = load_regular_run(waves_dir)
    ct = load_regular_run(control_dir)
    t = times(wv)
    t ≈ times(ct) || error("Waves and control output times differ")
    ΔU = zt(wv, "U") .- zt(ct, "U")
    if !isnothing(null_dir)
        nl = load_regular_run(null_dir; fields=("U",))
        U_null = zt(nl, "U")
        isnothing(quiescent_dir) || (U_null .-= zt(load_regular_run(quiescent_dir; fields=("U",)), "U"))
        ΔU .-= U_null
    end
    return ΔU, wv, ct
end

"""
    regular_ensemble(case, root, level, seeds; numerics="weno", Δt=0.02, Lx=3.2, Ly=3.2)

Per-seed FOV-window profiles of ΔU and Δ⟨u'w'⟩, their ensemble means and standard errors,
the ensemble-mean surface time series, the control anisotropy, and the shear ratio
R = −∂zΔU / ∂zuˢ.
"""
function regular_ensemble(case, root, level, seeds; numerics="weno", Δt=0.02, Lx=3.2, Ly=3.2, extra="")
    null = run_directory(root, case, level, "waves_null"; seed=0, Δt, numerics, Lx, Ly)
    quiescent = run_directory(root, case, level, "quiescent_control"; seed=0, Δt, numerics, Lx, Ly)
    isdir(quiescent) || (quiescent = nothing)
    isdir(null) || (null = nothing)
    profiles, Δuw_profiles, surfaces, anisotropies = [], [], [], []
    z = zf = t = uˢ = nothing
    tfov = k = 0.0
    for seed in seeds
        wv_dir = run_directory(root, case, level, "waves_turbulence"; seed, Δt, numerics, Lx, Ly, extra)
        ct_dir = run_directory(root, case, level, "turbulence_control"; seed, Δt, numerics, Lx, Ly, extra)
        ΔU, wv, ct = regular_residual(wv_dir, ct_dir, null, quiescent)
        t = times(wv); z = znodes_centers(wv); zf = znodes_faces(wv); tfov = t_fov(wv); k = k₀(wv)
        uˢ = stokes_profile(wv)
        w = fov_window(wv)
        push!(profiles, window_mean(ΔU, t, w...))
        m_wv, m_ct = regular_moments(wv), regular_moments(ct)
        push!(Δuw_profiles, window_mean(m_wv.uw .- m_ct.uw, t, w...))
        push!(surfaces, ΔU[end, :])
        ww_c = 0.5 .* (m_ct.ww[1:end-1, :] .+ m_ct.ww[2:end, :])
        push!(anisotropies, window_mean(m_ct.uu, t, w...) ./ max.(window_mean(ww_c, t, w...), 1e-12))
    end
    n = length(seeds)
    P, S, Σ = hcat(profiles...), hcat(Δuw_profiles...), hcat(surfaces...)
    mean_profile = vec(mean(P; dims=2))
    ∂zΔU = diff(mean_profile) ./ diff(z)
    ∂zuˢ = [2k * Float64(case.Uˢ₀) * exp(2k * zk) for zk in zf[2:end-1]]
    return (; case, level, seeds, extra, z, zf, t, k, uˢ, t_FOV = tfov,
              profiles = P, mean = mean_profile, stderr = vec(std(P; dims=2)) ./ sqrt(n),
              Δuw = S, Δuw_mean = vec(mean(S; dims=2)), Δuw_stderr = vec(std(S; dims=2)) ./ sqrt(n),
              surface = vec(mean(Σ; dims=2)), surface_stderr = vec(std(Σ; dims=2)) ./ sqrt(n),
              A = vec(mean(hcat(anisotropies...); dims=2)), R = -∂zΔU ./ ∂zuˢ)
end

function regular_report(r)
    kk = length(r.z)
    top = findall(zz -> zz > -1 / r.k, r.z)
    topf = findall(zz -> zz > -1 / r.k, r.zf[2:end-1])
    c = r.case
    println("\n", "="^78, "\n", "Case $(c.name) ($(c.family), ϵ = $(c.steepness), k = $(c.k) m⁻¹), $(r.level)$(isempty(r.extra) ? "" : " [" * r.extra * "]"), seeds $(r.seeds)", "\n", "="^78)
    @printf("  Uˢ₀ = %.2f mm/s, u_rf = %.2f mm/s, t_FOV = %.1f s\n", 1e3c.Uˢ₀, 1e3c.u_rf, r.t_FOV)
    @printf("  surface ΔU at the FOV: %.3f ± %.3f mm/s (per seed %s)  →  ΔU/Uˢ₀ = %.3f\n",
            1e3r.mean[kk], 1e3r.stderr[kk], join([@sprintf("%.2f", 1e3v) for v in r.profiles[kk, :]], ", "), r.mean[kk] / Float64(c.Uˢ₀))
    @printf("  mean over 0 > z > −1/k₀: ΔU = %.3f mm/s, uˢ = %.3f mm/s, ratio %.3f\n", 1e3mean(r.mean[top]), 1e3mean(r.uˢ[top]), mean(r.mean[top]) / mean(r.uˢ[top]))
    @printf("  shear ratio R = −∂zΔU/∂zuˢ over the Stokes layer: %.2f;  control anisotropy A = u'²/w'²: %.2f (R = 1 homogenized Lagrangian mean, R = A full quasi-equilibrium)\n",
            mean(r.R[topf]), mean(r.A[top]))
    @printf("  Δ⟨u'w'⟩ surface-adjacent: %.3e ± %.3e m²/s²\n", r.Δuw_mean[kk-1], r.Δuw_stderr[kk-1])
    println("  spin-up of the surface response (turbulence decays from the calibrated state at wave onset):")
    for f in (0.1, 0.25, 0.5, 0.75, 1.0)
        n = argmin(abs.(r.t .- f * r.t_FOV))
        @printf("    t = %5.1f s (%.2f t_FOV): surface ΔU = %7.3f ± %.3f mm/s, ΔU/Uˢ₀ = %.3f\n",
                r.t[n], f, 1e3r.surface[n], 1e3r.surface_stderr[n], r.surface[n] / Float64(c.Uˢ₀))
    end
    return nothing
end

"""
    regular_turbulence_report(dir)

Turbulence of a regular-wave control at t_FOV against the case targets, with suggested
per-component amplitude multipliers and the integral scale from the snapshot.
"""
function regular_turbulence_report(dir)
    run = load_regular_run(dir; fields=("U",))
    c = run_case(run)
    stats = load_statistics(dir)
    tf = t_fov(run)
    n = argmin(abs.(stats["t"] .- tf))
    target = (Float64(c.u_rms), Float64(c.v_rms), Float64(c.w_rms))
    measured = (stats["u_rms"][n], stats["v_rms"][n], stats["w_rms"][n])
    println("\n", "="^78, "\n", "Regular-wave turbulence control: $dir", "\n", "="^78)
    @printf("  volume rms at t_FOV = %.2f s: (%.4f, %.4f, %.4f) m/s; target (%.4f, %.4f, %.4f); ratio (%.2f, %.2f, %.2f)\n",
            stats["t"][n], measured..., target..., (measured ./ target)...)
    @printf("  volume rms at t = 0: (%.4f, %.4f, %.4f) ratio (%.2f, %.2f, %.2f); at t_end = %.2f s: (%.4f, %.4f, %.4f)\n",
            stats["u_rms"][1], stats["v_rms"][1], stats["w_rms"][1], stats["u_rms"][1] / target[1], stats["v_rms"][1] / target[2], stats["w_rms"][1] / target[3],
            stats["t"][end], stats["u_rms"][end], stats["v_rms"][end], stats["w_rms"][end])
    for f in (0.25, 0.5, 0.75)
        m = argmin(abs.(stats["t"] .- f * tf))
        @printf("  volume rms at %.2f t_FOV = %.1f s: (%.4f, %.4f, %.4f) ratio (%.2f, %.2f, %.2f)\n", f, stats["t"][m],
                stats["u_rms"][m], stats["v_rms"][m], stats["w_rms"][m], stats["u_rms"][m] / target[1], stats["v_rms"][m] / target[2], stats["w_rms"][m] / target[3])
    end
    ic = run.meta["ic_metadata"]
    amplitude = isnothing(ic) ? (NaN, NaN, NaN) : parse_amplitude(ic.amplitude)
    suggested = amplitude .* (target ./ measured)
    @printf("  amplitude multipliers (%.3f, %.3f, %.3f) → suggested (%.3f, %.3f, %.3f)\n", amplitude..., suggested...)
    if isfile(joinpath(dir, "snapshots.jld2"))
        u3, ts = load_snapshot(dir, "u", tf)
        Δx = run.meta["Lx"] / run.meta["Nx"]
        L = integral_scale_profile(u3, Δx)
        z = znodes_centers(run)
        @printf("  integral scale at t = %.2f s: L₁₁ = %.3f m at mid-depth, %.3f m in top cell (target %.3f m%s)\n",
                ts, L[argmin(abs.(z .+ Float64(c.h) / 2))], L[end], Float64(c.L), get(c, :L_assumed, false) ? ", assumed" : "")
    end
    return (; measured, target, suggested)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_key_value_args(ARGS)
    haskey(args, "report") && regular_turbulence_report(args["report"])
end
