#####
##### Ensemble-mean paired residual over turbulence seeds (campaign document, sections 12
##### and 16): the null-corrected post-packet ΔU(z) at the FOV for every seed, the ensemble
##### mean with its standard error, the depth-integrated transport, the Reynolds-stress
##### change, and the wake-age average, for one or more resolution levels / numerics.
#####
##### Usage: julia --project=. analysis/anti_stokes/ensemble_profile.jl case=1.D level=M0 seeds=1,2,3,4
#####            [level2=M1 seeds2=1,2,3,4 level3=M2 seeds3=1,...,8] [numerics=weno] [dt=0.02] [root=<dir>] [output=<png>]
#####

using CairoMakie
include("quick_checks.jl")

args = parse_key_value_args(ARGS)
root = getarg(args, "root", default_data_root())
numerics = getarg(args, "numerics", "weno")
Δt = getarg(args, "dt", 0.02)
x_topology = getarg(args, "x_topology", "periodic")
case = anti_stokes_case(getarg(args, "case", "1.D"))

levels = [(getarg(args, "level", "M0"), parse_seeds(getarg(args, "seeds", "1,2,3,4")))]
for n in 2:4
    haskey(args, "level$n") && push!(levels, (args["level$n"], parse_seeds(getarg(args, "seeds$n", getarg(args, "seeds", "1,2,3,4")))))
end

results = [ensemble(case, root, level, seeds; numerics, Δt, x_topology) for (level, seeds) in levels]

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
Label(fig[0, 1:3], "Case $(case.name) ensemble, null-corrected post-packet residual at the FOV (after − before), $numerics, Δt = $Δt s", fontsize=20)
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

output = get(args, "output", joinpath(figure_directory(), "ensemble_profile_$(case_dirname(case))_" * join(first.(levels), "_") * "_$(numerics)$(isempty(topology_tag(x_topology)) ? "" : "_" * topology_tag(x_topology)).png"))
save(output, fig)
@info "Saved $output"
