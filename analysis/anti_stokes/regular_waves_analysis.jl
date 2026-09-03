#####
##### Regular-wave cases: ensemble figures for one turbulence family (all its wavenumbers and
##### steepnesses) or a single case.
#####
#####   (1) null-corrected Eulerian change ΔU(z) at the FOV for every case, with −uˢ(z) (grey)
#####   (2) ΔU/Uˢ₀ against k₀z with the homogenized-Lagrangian-mean reference −e^{2k₀z} + const
#####   (3) spin-up: surface ΔU/Uˢ₀ against t/t_FOV
#####   (4) shear ratio R = −∂zΔU/∂zuˢ and anisotropy A at the FOV
#####
##### Usage: julia --project=. analysis/anti_stokes/regular_waves_analysis.jl family=2.A level=R1 seeds=1,2,3,4
#####        julia --project=. analysis/anti_stokes/regular_waves_analysis.jl case=3.A.2 level=R1 seeds=1,2,3,4
#####

using CairoMakie
include("regular_common.jl")

args = parse_key_value_args(ARGS)
level = getarg(args, "level", "R1")
seeds = parse.(Int, split(getarg(args, "seeds", "1,2,3,4"), ','))
numerics = getarg(args, "numerics", "weno")
Δt = getarg(args, "dt", 0.02)
Lx = getarg(args, "Lx", 3.2)
Ly = getarg(args, "Ly", 3.2)
root = getarg(args, "root", default_data_root())

case_names = if haskey(args, "case")
    [args["case"]]
else
    family = getarg(args, "family", "2.A")
    [name for (name, fam, _, _) in REGULAR_WAVE_TABLE if fam == family]
end
label = haskey(args, "case") ? args["case"] : getarg(args, "family", "2.A")

results = []
for name in case_names
    case = anti_stokes_case(name)
    try
        r = regular_ensemble(case, root, level, seeds; numerics, Δt, Lx, Ly)
        regular_report(r)
        push!(results, r)
    catch err
        @error "Case $name failed" exception=(err, catch_backtrace())
    end
end
isempty(results) && error("No cases could be analysed")

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1800, 1400))
Label(fig[0, 1:2], "Regular waves, $label at $level: null-corrected Eulerian change at the FOV, mean ± s.e. over $(length(seeds)) seeds", fontsize=22)
colors = Makie.wong_colors()

ax1 = Axis(fig[1, 1]; xlabel="ΔU (mm/s)", ylabel="k₀ z", title="(1) ΔU(z) at t_FOV; grey dashed: −uˢ(z)")
ax2 = Axis(fig[1, 2]; xlabel="ΔU / Uˢ₀", ylabel="k₀ z", title="(2) normalised; dashed: homogenized Lagrangian mean −e^{2k₀z} + ū")
ax3 = Axis(fig[2, 1]; xlabel="t / t_FOV", ylabel="surface ΔU / Uˢ₀", title="(3) spin-up of the surface anti-Stokes flow")
ax4 = Axis(fig[2, 2]; xlabel="ratio", ylabel="k₀ z", title="(4) R = −∂zΔU/∂zuˢ (solid) and A = u'²/w'² (dotted) at t_FOV")
for (j, r) in enumerate(results)
    c = r.case
    kz = r.k .* r.z
    Uˢ₀ = Float64(c.Uˢ₀)
    lbl = @sprintf("%s (k = %.1f, ϵ = %.2f, Uˢ₀ = %.1f mm/s)", c.name, c.k, c.steepness, 1e3Uˢ₀)
    band!(ax1, Point2f.(1e3 .* (r.mean .- r.stderr), kz), Point2f.(1e3 .* (r.mean .+ r.stderr), kz); color=(colors[mod1(j, 7)], 0.2))
    lines!(ax1, 1e3 .* r.mean, kz; color=colors[mod1(j, 7)], linewidth=3, label=lbl)
    lines!(ax1, -1e3 .* r.uˢ, kz; color=(:gray, 0.5), linestyle=:dash)
    lines!(ax2, r.mean ./ Uˢ₀, kz; color=colors[mod1(j, 7)], linewidth=3, label=c.name)
    band!(ax3, r.t ./ r.t_FOV, (r.surface .- r.surface_stderr) ./ Uˢ₀, (r.surface .+ r.surface_stderr) ./ Uˢ₀; color=(colors[mod1(j, 7)], 0.2))
    lines!(ax3, r.t ./ r.t_FOV, r.surface ./ Uˢ₀; color=colors[mod1(j, 7)], linewidth=2, label=c.name)
    kzf = r.k .* r.zf[2:end-1]
    shallow = kzf .> -2.0          # R is ill-defined where ∂zuˢ vanishes
    lines!(ax4, r.R[shallow], kzf[shallow]; color=colors[mod1(j, 7)], linewidth=3, label=c.name)
    lines!(ax4, r.A, kz; color=colors[mod1(j, 7)], linestyle=:dot)
end
r₁ = results[1]
kz₁ = r₁.k .* r₁.z
ū = -mean(exp.(2r₁.k .* r₁.z) .* Δz_centers(load_regular_run(run_directory(root, r₁.case, level, "waves_turbulence"; seed=first(seeds), Δt, numerics, Lx, Ly); fields=("U",)))) / Float64(r₁.case.h)
lines!(ax2, -exp.(2r₁.k .* r₁.z) .- ū, kz₁; color=:black, linestyle=:dash, label="−e^{2k₀z} − ⟨e^{2k₀z}⟩")
for ax in (ax1, ax2)
    vlines!(ax, [0]; color=(:black, 0.3))
    ylims!(ax, -4, 0)
    axislegend(ax; position=:rb, labelsize=12)
end
vlines!(ax3, [1]; color=(:black, 0.4), linestyle=:dot)
hlines!(ax3, [0, -1]; color=(:black, 0.3))
axislegend(ax3; position=:lb, labelsize=12)
vlines!(ax4, [0, 1]; color=(:black, 0.3))
xlims!(ax4, -1, 5)
ylims!(ax4, -4, 0)
axislegend(ax4; position=:rb, labelsize=12)

output = get(args, "output", joinpath(figure_directory(), "regular_waves_" * replace(label, "." => "") * "_$(level)_$(numerics).png"))
save(output, fig)
@info "Saved $output"
