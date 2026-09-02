#####
##### Ensemble-mean paired residual over turbulence seeds (campaign document, sections 12
##### and 16): the null-corrected post-packet ΔU(z) at the FOV for every seed, the ensemble
##### mean with its standard error, the depth-integrated transport, the Reynolds-stress
##### change, and the wake-age average, for one or more resolution levels / numerics.
#####
##### Usage: julia --project=. analysis/anti_stokes/ensemble_profile.jl level=M0 seeds=1,2,3,4
#####            [level2=M1 seeds2=1,2,3,4] [numerics=weno] [dt=0.02] [root=<dir>] [output=<png>]
#####

using CairoMakie
include("quick_checks.jl")

args = parse_key_value_args(ARGS)
root = getarg(args, "root", default_data_root())
numerics = getarg(args, "numerics", "weno")
Δt = getarg(args, "dt", 0.02)
case = anti_stokes_case("1.D")

parse_seeds(s) = parse.(Int, split(s, ','))

"""
    ensemble(level, seeds; numerics, Δt)

Per-seed FOV profiles (after − before) of the null-corrected ΔU and of Δ⟨u'w'⟩, plus the
wake-age (3–4σ₀) surface residual, for one level.
"""
function ensemble(level, seeds; numerics, Δt)
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
        before = window_mean(ΔU, t, 0, τ)[i, :]
        after  = window_mean(ΔU, t, 7τ, 8τ)[i, :]
        push!(profiles, after .- before)
        push!(transports, sum((after .- before) .* Δz))
        m_pk, m_ct = central_moments(pk), central_moments(ct)
        Δuw = m_pk.uw .- m_ct.uw
        push!(Δuw_profiles, window_mean(Δuw, t, 7τ, 8τ)[i, :] .- window_mean(Δuw, t, 0, τ)[i, :])
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

levels = [(getarg(args, "level", "M0"), parse_seeds(getarg(args, "seeds", "1,2,3,4")))]
haskey(args, "level2") && push!(levels, (args["level2"], parse_seeds(getarg(args, "seeds2", getarg(args, "seeds", "1,2,3,4")))))

results = [ensemble(level, seeds; numerics, Δt) for (level, seeds) in levels]

for r in results
    hr("Ensemble $(r.level), seeds $(r.seeds), numerics $numerics, Δt = $Δt")
    kk = length(r.z)
    @printf("  surface ΔU: mean %.3f ± %.3f mm/s (s.e.);  per seed: %s\n", 1e3r.mean[kk], 1e3r.stderr[kk],
            join([@sprintf("%.2f", 1e3v) for v in r.profiles[kk, :]], ", "))
    kmin = argmin(r.mean)
    @printf("  minimum ΔU: %.3f ± %.3f mm/s at k₀z = %.2f\n", 1e3r.mean[kmin], 1e3r.stderr[kmin], r.k * r.z[kmin])
    kmax = argmax(r.mean)
    @printf("  maximum ΔU: %.3f ± %.3f mm/s at k₀z = %.2f\n", 1e3r.mean[kmax], 1e3r.stderr[kmax], r.k * r.z[kmax])
    @printf("  wake-age (3–4σ₀) surface ΔU at t_stop: mean %.3f mm/s, per seed %s\n", 1e3mean(r.wakes),
            join([@sprintf("%.2f", 1e3v) for v in r.wakes], ", "))
    @printf("  depth integral ∫ΔU dz per seed: %s m²/s\n", join([@sprintf("%.2e", v) for v in r.transports], ", "))
    @printf("  Δ⟨u'w'⟩ surface-adjacent cell: %.3e ± %.3e m²/s²\n", r.Δuw_mean[kk-1], r.Δuw_stderr[kk-1])
    @printf("  wake-age composite (age 2–4 τ₀) surface ΔU: %.3f ± %.3f mm/s; min %.3f mm/s at k₀z = %.2f; per seed %s\n",
            1e3r.composite_mean[kk], 1e3r.composite_stderr[kk], 1e3minimum(r.composite_mean),
            r.k * r.z[argmin(r.composite_mean)], join([@sprintf("%.2f", 1e3v) for v in r.composites[kk, :]], ", "))
    # Signal-to-noise: depth-averaged over the top Stokes depth
    top = findall(zz -> zz > -1 / case.k, r.z)
    @printf("  mean over 0 > z > −1/k₀: ΔU = %.3f ± %.3f mm/s  (|mean|/s.e. = %.1f)\n",
            1e3mean(r.mean[top]), 1e3mean(r.stderr[top]), abs(mean(r.mean[top])) / max(mean(r.stderr[top]), 1e-12))
end

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1500, 800))
Label(fig[0, 1:3], "Case 1.D ensemble, null-corrected post-packet residual at the FOV (after − before), $numerics, Δt = $Δt s", fontsize=20)
colors = Makie.wong_colors()

ax1 = Axis(fig[1, 1]; xlabel="ΔU (mm/s)", ylabel="k₀ z", title="ΔU(z): seeds (thin) and ensemble mean ± s.e.")
ax2 = Axis(fig[1, 2]; xlabel="Δ⟨u'w'⟩ (10⁻⁶ m²/s²)", ylabel="k₀ z", title="Reynolds-stress change")
ax3 = Axis(fig[1, 3]; xlabel="mm/s", ylabel="k₀ z", title="wake-age composite (age 2–4 τ₀), mean ± s.e.")
for (j, r) in enumerate(results)
    kz = r.k .* r.z
    for s in axes(r.profiles, 2)
        lines!(ax1, 1e3 .* r.profiles[:, s], kz; color=(colors[j], 0.35), linewidth=1)
        lines!(ax2, 1e6 .* r.Δuw[:, s], kz; color=(colors[j], 0.35), linewidth=1)
    end
    band!(ax1, Point2f.(1e3 .* (r.mean .- r.stderr), kz), Point2f.(1e3 .* (r.mean .+ r.stderr), kz); color=(colors[j], 0.25))
    lines!(ax1, 1e3 .* r.mean, kz; color=colors[j], linewidth=3, label="$(r.level) mean (n = $(length(r.seeds)))")
    lines!(ax2, 1e6 .* r.Δuw_mean, kz; color=colors[j], linewidth=3, label=r.level)
    for s in axes(r.composites, 2)
        lines!(ax3, 1e3 .* r.composites[:, s], kz; color=(colors[j], 0.35), linewidth=1)
    end
    band!(ax3, Point2f.(1e3 .* (r.composite_mean .- r.composite_stderr), kz),
          Point2f.(1e3 .* (r.composite_mean .+ r.composite_stderr), kz); color=(colors[j], 0.25))
    lines!(ax3, 1e3 .* r.composite_mean, kz; color=colors[j], linewidth=3, label="$(r.level) composite")
end
# Reference: −uˢ profile shape scaled to the surface value of the first composite mean
r = results[1]
lines!(ax3, 1e3 .* r.composite_mean[end] .* exp.(2case.k .* r.z), r.k .* r.z; color=:black, linestyle=:dash, label="surface value × e^{2k₀z}")
for ax in (ax1, ax2, ax3)
    vlines!(ax, [0]; color=(:black, 0.3))
    ylims!(ax, -4, 0)
    axislegend(ax; position=:rb)
end

output = get(args, "output", joinpath(figure_directory(), "ensemble_profile_" * join(first.(levels), "_") * "_$(numerics).png"))
save(output, fig)
@info "Saved $output"
