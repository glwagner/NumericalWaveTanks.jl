#####
##### Makie-free acceptance reports (campaign document, section 16). Each function
##### prints a short report and returns the key numbers as a NamedTuple.
#####
##### Usage from the command line:
#####   julia --project=. analysis/anti_stokes/quick_checks.jl null=<dir> [control=<dir> packet=<dir> quiescent=<dir>]
#####

include("common.jl")

hr(title) = println("\n", "="^78, "\n", title, "\n", "="^78)

"""
    quiescent_report(dir)

The quiescent control should stay at rest apart from roundoff.
"""
function quiescent_report(dir)
    run = load_run(dir; fields=("U", "V", "W"))
    U, V, W = xzt(run, "U"), xzt(run, "V"), xzt(run, "W")
    hr("Quiescent control: $dir")
    @printf("  max |⟨u⟩|, |⟨v⟩|, |⟨w⟩| over the run: %.3e, %.3e, %.3e m/s (roundoff expected)\n",
            maximum(abs, U), maximum(abs, V), maximum(abs, W))
    return (; max_u = maximum(abs, U), max_v = maximum(abs, V), max_w = maximum(abs, W))
end

"""
    packet_null_report(dir)

Packet-only null: trajectory, Eulerian residual at the FOV before and after the packet,
return-flow structure at the peak, and momentum drift.
"""
function packet_null_report(dir)
    run = load_run(dir; fields=("U", "W"))
    p = run_packet(run)
    t, x, z, Δz = times(run), xnodes_faces(run), znodes_centers(run), Δz_centers(run)
    τ, tp, i = τ₀(run), t_peak(run), fov_index(run)
    stats = load_statistics(dir)
    hr("Packet-only null: $dir")

    # 1. Packet trajectory at the FOV
    analytic = @. p.Uˢ₀ * exp(-((stats["t"] - tp) / τ)^2)
    trajectory_error = maximum(abs, stats["uˢ_fov"] .- analytic) / p.Uˢ₀
    @printf("  trajectory: max |uˢ(x_FOV, 0, t) − Uˢ₀ e^{-(t-t_peak)²/τ₀²}| / Uˢ₀ = %.2e\n", trajectory_error)

    # 2. Eulerian residual at the FOV
    UE = eulerian_U(run)
    w = windows(run)
    before = window_mean(UE, t, w.before[1], min(w.before[2], t[end]))
    residual_before = maximum(abs, before[i, :])
    @printf("  |⟨u^E⟩| at FOV, before window (max over z): %.3f mm/s\n", 1e3 * residual_before)
    complete = t[end] >= w.after[1] - 1e-6
    residual_after = NaN
    if complete
        after = window_mean(UE, t, w.after...)
        residual_after = maximum(abs, after[i, :])
        @printf("  |⟨u^E⟩| at FOV, after window  (max over z): %.3f mm/s   [target < 0.1 mm/s]\n", 1e3 * residual_after)
        @printf("  surface ⟨u^E⟩ at FOV, after − before: %.3f mm/s\n", 1e3 * (after[i, end] - before[i, end]))
    else
        @printf("  run ends at t = %.2f s before the after window: skipped\n", t[end])
    end

    # Far-field residual at the peak (packet centred on the FOV, x = 0 is 6 m away)
    n_peak = nearest_index(t, tp)
    abs(t[n_peak] - tp) <= 0.51 * (t[2] - t[1]) || @printf("  (run does not reach t_peak; using t = %.2f s for the following)\n", t[n_peak])
    far = UE[1, :, n_peak]
    @printf("  |⟨u^E⟩| at x = 0 (far field) at t_peak (max over z): %.3f mm/s\n", 1e3 * maximum(abs, far))

    # 3. Return flow follows the packet
    surface = UE[:, end, n_peak]
    i_min = argmin(surface)
    @printf("  t_peak: surface ⟨u^E⟩ at FOV = %.3f mm/s; min over x = %.3f mm/s at x = %.3f m (x_c = %.3f m)\n",
            1e3 * surface[i], 1e3 * surface[i_min], x[i_min], packet_center(t[n_peak], p))
    transport_E = sum(UE[i, :, n_peak] .* Δz)
    transport_S = sum([uˢ(x[i], 0, zk, t[n_peak], p) for zk in z] .* Δz)
    @printf("  t_peak: ∫⟨u^E⟩dz at FOV = %.3e m²/s, ∫uˢ dz = %.3e m²/s, ratio = %.3f (−1 = complete local return flow)\n",
            transport_E, transport_S, transport_E / transport_S)

    # 4. Momentum
    @printf("  domain-mean u^L: %.3e → %.3e m/s (change %.3e m/s)\n",
            stats["u_mean"][1], stats["u_mean"][end], stats["u_mean"][end] - stats["u_mean"][1])
    @printf("  max |w| over the run: %.3e m/s\n", maximum(stats["w_max"]))

    return (; trajectory_error, residual_before, residual_after, transport_ratio = transport_E / transport_S,
              far_field = maximum(abs, far), momentum_drift = stats["u_mean"][end] - stats["u_mean"][1])
end

"""
    null_convergence_report(dir₁, dir₂)

Compare the Eulerian response of two packet-only nulls with identical output times
(for example, Δt and Δt/2).
"""
function null_convergence_report(dir₁, dir₂)
    r₁ = load_run(dir₁; fields=("U",))
    r₂ = load_run(dir₂; fields=("U",))
    hr("Null convergence: $dir₁ vs $dir₂")
    t = times(r₁)
    if !(length(t) == length(times(r₂)) && t ≈ times(r₂)) || size(xzt(r₁, "U")) != size(xzt(r₂, "U"))
        println("  output grids differ; skipping pointwise comparison")
        return nothing
    end
    U₁, U₂ = eulerian_U(r₁), eulerian_U(r₂)
    τ, i = τ₀(r₁), fov_index(r₁)
    n_peak = nearest_index(t, t_peak(r₁))
    peak_change = maximum(abs, U₁[:, :, n_peak] .- U₂[:, :, n_peak]) / maximum(abs, U₁[:, :, n_peak])
    a₁, a₂ = window_mean(U₁, t, after_window(r₁)...)[i, :], window_mean(U₂, t, after_window(r₁)...)[i, :]
    @printf("  Eulerian response at t_peak: max relative change %.2e\n", peak_change)
    @printf("  post-packet FOV residual: %.4f mm/s vs %.4f mm/s (change %.4f mm/s)\n",
            1e3 * maximum(abs, a₁), 1e3 * maximum(abs, a₂), 1e3 * maximum(abs, a₁ .- a₂))
    return (; peak_change, residual₁ = maximum(abs, a₁), residual₂ = maximum(abs, a₂))
end

"""
    turbulence_report(dir)

Turbulence statistics of a no-wave control: rms components at t_peak against the
case targets, near-surface anisotropy, and the streamwise integral scale.
"""
function turbulence_report(dir)
    run = load_run(dir)
    c = run_case(run)
    t, z = times(run), znodes_centers(run)
    tp = t_peak(run)
    stats = load_statistics(dir)
    hr("Turbulence control: $dir")

    n = nearest_index(stats["t"], tp)
    target = (Float64(c.u_rms), Float64(c.v_rms), Float64(c.w_rms))
    measured = (stats["u_rms"][n], stats["v_rms"][n], stats["w_rms"][n])
    @printf("  volume rms at t = %.2f s: (%.4f, %.4f, %.4f) m/s; target (%.4f, %.4f, %.4f); ratio (%.2f, %.2f, %.2f)\n",
            stats["t"][n], measured..., target..., (measured ./ target)...)
    @printf("  volume rms at t = 0:      (%.4f, %.4f, %.4f) m/s\n", stats["u_rms"][1], stats["v_rms"][1], stats["w_rms"][1])
    @printf("  volume rms at t = %.2f s: (%.4f, %.4f, %.4f) m/s\n", stats["t"][end], stats["u_rms"][end], stats["v_rms"][end], stats["w_rms"][end])

    ic_meta = run.meta["ic_metadata"]
    amplitude = isnothing(ic_meta) ? (NaN, NaN, NaN) : parse_amplitude(ic_meta.amplitude)
    suggested = amplitude .* (target ./ measured)
    @printf("  amplitude multipliers (%.3f, %.3f, %.3f) → suggested (%.3f, %.3f, %.3f) to match rms at t_peak\n",
            amplitude..., suggested...)

    m = central_moments(run)
    n_avg = nearest_index(t, tp)
    uu = dropdims(mean(m.uu[:, :, n_avg]; dims=1); dims=1)
    ww = dropdims(mean(m.ww[:, :, n_avg]; dims=1); dims=1)
    wwc = 0.5 .* (ww[1:end-1] .+ ww[2:end])
    println("  rms profiles at t_peak (x-averaged):")
    println("     z [m]     u_rms    w_rms    u'²/w'²")
    for k in reverse(eachindex(z))[1:min(8, end)]
        @printf("    %7.4f   %.4f   %.4f   %6.2f\n", z[k], sqrt(max(uu[k], 0)), sqrt(max(wwc[k], 0)), uu[k] / max(wwc[k], 1e-12))
    end
    k_mid = nearest_index(z, -0.2)
    @printf("    mid-depth (z = %.3f): u_rms %.4f, w_rms %.4f, u'²/w'² %.2f\n",
            z[k_mid], sqrt(max(uu[k_mid], 0)), sqrt(max(wwc[k_mid], 0)), uu[k_mid] / max(wwc[k_mid], 1e-12))

    L = NaN
    if isfile(joinpath(dir, "snapshots.jld2"))
        u3, ts = load_snapshot(dir, "u", tp)
        Δx = run.meta["Lx"] / run.meta["Nx"]
        Lprof = integral_scale_profile(u3, Δx)
        L = Lprof[k_mid]
        @printf("  streamwise integral scale from snapshot at t = %.2f s: L₁₁ = %.3f m at mid-depth, %.3f m in top cell (target %.3f m)\n",
                ts, L, Lprof[end], Float64(c.L))
    end

    return (; measured, target, suggested_amplitude = suggested, L)
end

"""
    pair_report(packet_dir, control_dir, null_dir=nothing, quiescent_dir=nothing)

Paired before/after residual ΔU(z) at the observation plane with optional
packet-null correction, plus the Reynolds-stress change and the depth integral.
"""
function pair_report(packet_dir, control_dir, null_dir=nothing, quiescent_dir=nothing)
    ΔU, pk, ct = paired_residual(packet_dir, control_dir, null_dir, quiescent_dir)
    t, z, Δz = times(pk), znodes_centers(pk), Δz_centers(pk)
    τ, i, k = τ₀(pk), fov_index(pk), k₀(pk)
    hr("Paired residual at the FOV: $packet_dir")
    isnothing(null_dir) && println("  (no packet-null correction)")

    w = windows(pk)
    before = window_mean(ΔU, t, w.before...)[i, :]
    after  = window_mean(ΔU, t, w.after...)[i, :]
    profile = after .- before
    kmin, kmax = argmin(profile), argmax(profile)
    @printf("  surface ΔU = %.3f mm/s; min %.3f mm/s at k₀z = %.2f; max %.3f mm/s at k₀z = %.2f\n",
            1e3 * profile[end], 1e3 * profile[kmin], k * z[kmin], 1e3 * profile[kmax], k * z[kmax])
    I = depth_integral(profile, Δz)
    @printf("  ∫ΔU dz = %.3e m²/s (positive lobe %.3e, negative lobe %.3e; |total| / (|+| + |−|) = %.3f)\n",
            I.total, I.positive, I.negative, abs(I.total) / max(I.positive - I.negative, 1e-30))
    @printf("  pre-packet |ΔU| at FOV (max over z): %.3f mm/s  (leakage / noise floor)\n", 1e3 * maximum(abs, before))

    # Sampling noise proxy: rms of ΔU along x far from the packet in the after window,
    # excluding the wake (x > x_c − 2σ₀ is the packet/wake region at the end)
    p = run_packet(pk)
    x = xnodes_faces(pk)
    after_xz = window_mean(ΔU, t, w.after...)
    wake = [xi < packet_center(w.after[1], p) - 2p.σ₀ && xi > p.x_FOV - 2p.σ₀ for xi in x]  # wake region
    @printf("  wake-averaged surface ΔU (%.1f m < x < %.1f m): %.3f mm/s, x-std %.3f mm/s\n",
            p.x_FOV - 2p.σ₀, packet_center(w.after[1], p) - 2p.σ₀,
            1e3 * mean(after_xz[wake, end]), 1e3 * std(after_xz[wake, end]))

    # Wake-age composite: the low-noise estimate
    ages, C, N = wake_age_composite(ΔU, x, t, p; age_edges=default_age_edges(τ))
    println("  wake-age composite of surface ΔU (all x, t with the same time since passage):")
    for (a₀, a₁) in ((-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 6))
        prof_a = composite_profile(ages, C, τ, a₀ + 0.125, a₁ - 0.125)
        Ia = depth_integral(prof_a, Δz)
        @printf("    age %2d–%2d τ₀: surface %7.3f mm/s, min %7.3f mm/s at k₀z = %5.2f, ∫dz = %9.2e m²/s\n",
                a₀, a₁, 1e3 * prof_a[end], 1e3 * minimum(prof_a), k * z[argmin(prof_a)], Ia.total)
    end
    composite_after = composite_profile(ages, C, τ, 2.125, 3.875)

    m_pk, m_ct = central_moments(pk), central_moments(ct)
    Δuw = m_pk.uw .- m_ct.uw
    Δuw_profile = window_mean(Δuw, t, w.after...)[i, :] .- window_mean(Δuw, t, w.before...)[i, :]
    Δuw_passage = window_mean(Δuw, t, w.passage...)[i, :]
    @printf("  Δ⟨u'w'⟩ during passage (t_peak ± τ₀): max |Δu'w'| = %.3e m²/s²; after − before: %.3e m²/s²\n",
            maximum(abs, Δuw_passage), maximum(abs, Δuw_profile))

    # Quasi-equilibrium ratio R = −∂z ΔU / ∂z uˢ (peak Stokes shear) vs anisotropy A = u'²/w'²
    zf = znodes_faces(pk)
    ∂zΔU = diff(profile) ./ diff(z)
    ∂zuˢ = [2p.k * p.Uˢ₀ * exp(2p.k * zk) for zk in zf[2:end-1]]
    R = -∂zΔU ./ ∂zuˢ
    uu_ct = window_mean(m_ct.uu, t, w.after...)[i, :]
    ww_ct = window_mean(m_ct.ww, t, w.after...)[i, :]
    A = uu_ct[2:end] ./ max.(ww_ct[2:end-1], 1e-12)
    println("  top levels:  k₀z     ΔU [mm/s]   Δu'w' [mm²/s²]    R=−∂zΔU/∂zuˢ   A=u'²/w'²")
    for kk in reverse(eachindex(z))[1:min(10, end)]
        kf = kk - 1
        Rk = kf >= 1 ? R[kf] : NaN
        Ak = kf >= 1 ? A[kf] : NaN
        @printf("             %6.2f   %9.3f   %13.4f   %12.3f   %10.2f\n", k * z[kk], 1e3 * profile[kk], 1e6 * Δuw_profile[kk], Rk, Ak)
    end

    return (; z, profile, composite_after, ages, composite = C, Δuw_profile, integral = I, noise_floor = maximum(abs, before))
end

#####
##### Ensemble over seeds (used by ensemble_profile.jl and compare_cases.jl)
#####

parse_seeds(s) = parse.(Int, split(s, ','))

"""
    ensemble(case, root, level, seeds; numerics="weno", Δt=0.02)

Per-seed FOV profiles (after − before) of the null-corrected ΔU and of Δ⟨u'w'⟩, plus the
wake-age (3–4σ₀) surface residual, for one level.
"""
function ensemble(case, root, level, seeds; numerics="weno", Δt=0.02)
    null = run_directory(root, case, level, "packet_null"; seed=0, Δt, numerics)
    quiescent = run_directory(root, case, level, "quiescent_control"; seed=0, Δt, numerics)
    isdir(quiescent) || (quiescent = nothing)
    profiles, Δuw_profiles, wakes, transports, composites = [], [], Float64[], Float64[], []
    z, k = nothing, nothing
    for seed in seeds
        pk_dir = run_directory(root, case, level, "packet_turbulence"; seed, Δt, numerics)
        ct_dir = run_directory(root, case, level, "turbulence_control"; seed, Δt, numerics)
        ΔU, pk, ct = paired_residual(pk_dir, ct_dir, null, quiescent)
        t, x, Δz = times(pk), xnodes_faces(pk), Δz_centers(pk)
        z, k = znodes_centers(pk), k₀(pk)
        τ, i, p = τ₀(pk), fov_index(pk), run_packet(pk)
        w = windows(pk)
        before = window_mean(ΔU, t, w.before...)[i, :]
        after  = window_mean(ΔU, t, w.after...)[i, :]
        push!(profiles, after .- before)
        push!(transports, sum((after .- before) .* Δz))
        m_pk, m_ct = central_moments(pk), central_moments(ct)
        Δuw = m_pk.uw .- m_ct.uw
        push!(Δuw_profiles, window_mean(Δuw, t, w.after...)[i, :] .- window_mean(Δuw, t, w.before...)[i, :])
        xc_end = packet_center(t[end], p)   # outside @. so the NamedTuple p is not broadcast
        age = @. mod(xc_end - x, pk.meta["Lx"])
        wake = findall(a -> 3p.σ₀ <= a <= 4p.σ₀, age)
        push!(wakes, mean(ΔU[wake, end, end]) - before[end])
        ages, C, _ = wake_age_composite(ΔU, x, t, p; age_edges=default_age_edges(τ))
        push!(composites, composite_profile(ages, C, τ, 2.125, 3.875))
    end
    P = hcat(profiles...)
    S = hcat(Δuw_profiles...)
    W = hcat(composites...)
    n = length(seeds)
    return (; level, seeds, z, k, profiles = P, Δuw = S, composites = W,
              mean = vec(mean(P; dims=2)), stderr = vec(std(P; dims=2)) ./ sqrt(n),
              composite_mean = vec(mean(W; dims=2)), composite_stderr = vec(std(W; dims=2)) ./ sqrt(n),
              Δuw_mean = vec(mean(S; dims=2)), Δuw_stderr = vec(std(S; dims=2)) ./ sqrt(n),
              wakes, transports)
end


if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_key_value_args(ARGS)
    haskey(args, "quiescent") && quiescent_report(args["quiescent"])
    haskey(args, "null") && packet_null_report(args["null"])
    haskey(args, "null2") && null_convergence_report(args["null"], args["null2"])
    haskey(args, "control") && turbulence_report(args["control"])
    if haskey(args, "packet") && haskey(args, "control")
        pair_report(args["packet"], args["control"], get(args, "null", nothing), get(args, "quiescent", nothing))
    end
end
