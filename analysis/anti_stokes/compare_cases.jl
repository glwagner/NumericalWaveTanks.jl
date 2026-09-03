#####
##### Cross-case comparison of the wave-group cases (campaign document, 13.1–13.2):
#####
#####   (1) wake-age composite ΔU(z) for every case, absolute
#####   (2) the same normalised by the surface Stokes drift Uˢ₀ against k₀z
#####   (3) surface composite ΔU against turbulent kinetic energy e, with the
#####       (k₀aₚ)² scaling of the steepness pair 1.C.1 / 1.C.2
#####   (4) composite surface ΔU versus age (spin-up) for every case
#####
##### Usage: julia --project=. analysis/anti_stokes/compare_cases.jl level=M2 cases=1.A,1.B,1.C.1,1.C.2,1.D
#####            seeds=1,2,3,4 [seeds_1.D=1,2,3,4,5,6,7,8] [numerics=weno] [dt=0.02] [root=<dir>]
#####

using CairoMakie
include("quick_checks.jl")

args = parse_key_value_args(ARGS)
level = getarg(args, "level", "M2")
case_names = split(getarg(args, "cases", "1.A,1.B,1.C.1,1.C.2,1.D"), ',')
default_seeds = getarg(args, "seeds", "1,2,3,4")
numerics = getarg(args, "numerics", "weno")
Δt = getarg(args, "dt", 0.02)
root = getarg(args, "root", default_data_root())

"""
    spinup_curve(case, root, level, seeds; numerics, Δt)

Ensemble-mean composite surface ΔU as a function of age.
"""
function spinup_curve(case, root, level, seeds; numerics, Δt)
    curves = []
    ages = nothing
    null = run_directory(root, case, level, "packet_null"; seed=0, Δt, numerics)
    quiescent = run_directory(root, case, level, "quiescent_control"; seed=0, Δt, numerics)
    isdir(quiescent) || (quiescent = nothing)
    for seed in seeds
        pk_dir = run_directory(root, case, level, "packet_turbulence"; seed, Δt, numerics)
        ct_dir = run_directory(root, case, level, "turbulence_control"; seed, Δt, numerics)
        ΔU, pk, _ = paired_residual(pk_dir, ct_dir, null, quiescent)
        τ = τ₀(pk)
        ages, C, _ = wake_age_composite(ΔU, xnodes_faces(pk), times(pk), run_packet(pk); age_edges=default_age_edges(τ))
        push!(curves, C[end, :])
        ages = ages ./ τ
    end
    M = hcat(curves...)
    return ages, vec(mean(M; dims=2)), vec(std(M; dims=2)) ./ sqrt(length(seeds))
end

results = []
for name in case_names
    case = anti_stokes_case(String(name))
    seeds = parse_seeds(getarg(args, "seeds_$name", default_seeds))
    try
        r = ensemble(case, root, level, seeds; numerics, Δt)
        ages, curve, curve_err = spinup_curve(case, root, level, seeds; numerics, Δt)
        push!(results, (; case, r, ages, curve, curve_err))
    catch err
        @error "Case $name failed" exception=(err, catch_backtrace())
    end
end

hr("Cross-case comparison at $level ($numerics, Δt = $Δt)")
println("  case     ϵ     Uˢ₀ (mm/s)   e (cm²/s²)   L (cm)   u²/w²   surface ΔU (mm/s)   ΔU/Uˢ₀   ΔU/(Uˢ₀ e^{1/2})   lobe (mm/s) at k₀z")
for (; case, r) in results
    kk = length(r.z)
    A = (Float64(case.u_rms) / Float64(case.w_rms))^2
    kmax = argmax(r.composite_mean)
    @printf("  %-6s  %.2f   %6.1f      %6.2f      %5.1f   %5.2f   %7.2f ± %5.2f    %6.3f     %8.3f        %5.2f at %5.2f\n",
            case.name, case.steepness, 1e3case.Uˢ₀, 1e4case.e, 1e2case.L, A,
            1e3r.composite_mean[kk], 1e3r.composite_stderr[kk], r.composite_mean[kk] / case.Uˢ₀,
            r.composite_mean[kk] / (case.Uˢ₀ * sqrt(Float64(case.e))), 1e3r.composite_mean[kmax], r.k * r.z[kmax])
end

# Steepness pair
i1 = findfirst(x -> x.case.name == "1.C.1", results)
i2 = findfirst(x -> x.case.name == "1.C.2", results)
if !isnothing(i1) && !isnothing(i2)
    r1, r2 = results[i1].r, results[i2].r
    ratio = r2.composite_mean[end] / r1.composite_mean[end]
    err = ratio * sqrt((r2.composite_stderr[end] / r2.composite_mean[end])^2 + (r1.composite_stderr[end] / r1.composite_mean[end])^2)
    ϵ1, ϵ2 = Float64(results[i1].case.steepness), Float64(results[i2].case.steepness)
    @printf("  steepness pair: surface ΔU(1.C.2) / ΔU(1.C.1) = %.2f ± %.2f;  (ϵ₂/ϵ₁)² = %.2f;  Uˢ₀ ratio = %.2f\n",
            ratio, err, (ϵ2 / ϵ1)^2, Float64(results[i2].case.Uˢ₀) / Float64(results[i1].case.Uˢ₀))
end

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1800, 1400))
Label(fig[0, 1:2], "Wave-group cases at $level: null-corrected wake-age composites (age 2–4 τ₀), mean ± s.e.", fontsize=22)
colors = Makie.wong_colors()

ax1 = Axis(fig[1, 1]; xlabel="ΔU (mm/s)", ylabel="k₀ z", title="(1) composite ΔU(z)")
ax2 = Axis(fig[1, 2]; xlabel="ΔU / Uˢ₀", ylabel="k₀ z", title="(2) normalised by the surface Stokes drift")
ax3 = Axis(fig[2, 1]; xlabel="e (cm²/s²)", ylabel="surface ΔU (mm/s)", title="(3) surface residual vs turbulent kinetic energy")
ax4 = Axis(fig[2, 2]; xlabel="age (τ₀)", ylabel="surface ΔU (mm/s)", title="(4) spin-up: composite surface ΔU vs age")
for (j, (; case, r, ages, curve, curve_err)) in enumerate(results)
    kz = r.k .* r.z
    label = "$(case.name) (ϵ = $(case.steepness), e = $(round(1e4case.e, digits=2)) cm²/s²)"
    band!(ax1, Point2f.(1e3 .* (r.composite_mean .- r.composite_stderr), kz), Point2f.(1e3 .* (r.composite_mean .+ r.composite_stderr), kz); color=(colors[j], 0.2))
    lines!(ax1, 1e3 .* r.composite_mean, kz; color=colors[j], linewidth=3, label)
    lines!(ax2, r.composite_mean ./ Float64(case.Uˢ₀), kz; color=colors[j], linewidth=3, label=case.name)
    scatter!(ax3, [1e4case.e], [1e3r.composite_mean[end]]; color=colors[j], markersize=18, label=case.name)
    errorbars!(ax3, [1e4case.e], [1e3r.composite_mean[end]], [1e3r.composite_stderr[end]]; color=colors[j], whiskerwidth=10)
    band!(ax4, ages, 1e3 .* (curve .- curve_err), 1e3 .* (curve .+ curve_err); color=(colors[j], 0.2))
    lines!(ax4, ages, 1e3 .* curve; color=colors[j], linewidth=3, label=case.name)
end
lines!(ax2, -exp.(2 .* range(-4, 0, length=50) ./ 2) .* 0.25, range(-4, 0, length=50); color=:black, linestyle=:dash, label="−0.25 e^{k₀z}")
for ax in (ax1, ax2)
    vlines!(ax, [0]; color=(:black, 0.3))
    ylims!(ax, -4, 0)
    axislegend(ax; position=:rb, labelsize=13)
end
hlines!(ax3, [0]; color=(:black, 0.3))
axislegend(ax3; position=:rt, labelsize=13)
vlines!(ax4, [0]; color=(:black, 0.3), linestyle=:dot)
hlines!(ax4, [0]; color=(:black, 0.3))
axislegend(ax4; position=:rt, labelsize=13)

output = get(args, "output", joinpath(figure_directory(), "compare_cases_$(level)_$(numerics).png"))
save(output, fig)
@info "Saved $output"
