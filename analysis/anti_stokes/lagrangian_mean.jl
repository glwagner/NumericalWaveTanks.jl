#####
##### Is the Lagrangian-mean flow homogenized beneath the packet?
#####
##### Wake-age composites of the null-corrected Eulerian response ΔU, the prescribed Stokes
##### drift uˢ at the same packet coordinate, and their sum, the turbulence-induced change of
##### the Lagrangian mean uᴸ = uˢ + ΔU, as profiles beneath the packet (ages −2 … +2 τ₀) and as
##### shear ratios. The quasi-equilibrium relation ∂z uᴱ = −(u'²/w'²) ∂z uˢ predicts a
##### homogenized Lagrangian mean only for isotropic turbulence; complete spin-up would give
##### R = −∂zΔU / ∂zuˢ → A = u'²/w'².
#####
##### Usage: julia --project=. analysis/anti_stokes/lagrangian_mean.jl case=1.D level=M2 seeds=1,...,8
#####            [numerics=weno] [dt=0.02] [root=<dir>] [output=<png>]
#####

using CairoMakie
include("quick_checks.jl")

args = parse_key_value_args(ARGS)
case = anti_stokes_case(getarg(args, "case", "1.D"))
level = getarg(args, "level", "M2")
seeds = parse_seeds(getarg(args, "seeds", "1,2,3,4"))
numerics = getarg(args, "numerics", "weno")
Δt = getarg(args, "dt", 0.02)
root = getarg(args, "root", default_data_root())
x_topology = getarg(args, "x_topology", "periodic")

null = run_directory(root, case, level, "packet_null"; seed=0, Δt, numerics, x_topology)
quiescent = run_directory(root, case, level, "quiescent_control"; seed=0, Δt, numerics, x_topology)
isdir(quiescent) || (quiescent = nothing)

composites, anisotropies = [], []
ages = z = zf = Δz = p = τ = k = nothing
for seed in seeds
    pk_dir = run_directory(root, case, level, "packet_turbulence"; seed, Δt, numerics, x_topology)
    ct_dir = run_directory(root, case, level, "turbulence_control"; seed, Δt, numerics, x_topology)
    ΔU, pk, ct = paired_residual(pk_dir, ct_dir, null, quiescent)
    global p, τ, k = run_packet(pk), τ₀(pk), k₀(pk)
    global z, zf, Δz = znodes_centers(pk), znodes_faces(pk), Δz_centers(pk)
    t, x = times(pk), xnodes_faces(pk)
    a, C, _ = wake_age_composite(ΔU, x, t, p; age_edges=default_age_edges(τ))
    global ages = a
    push!(composites, C)
    # anisotropy of the control during the passage window, x-averaged
    m = central_moments(ct)
    uu = vec(mean(window_mean(m.uu, t, passage_window(pk)...); dims=1))
    ww = vec(mean(window_mean(m.ww, t, passage_window(pk)...); dims=1))
    push!(anisotropies, uu ./ max.(0.5 .* (ww[1:end-1] .+ ww[2:end]), 1e-12))
end

n = length(seeds)
ΔU_mean = mean(composites)                      # (Nz, Na)
ΔU_err = std(composites) ./ sqrt(n)
A_mean = mean(anisotropies)

# Stokes drift at the same packet coordinate: age a ↔ ξ = −cᵍ a
Uˢ = [uˢ(-p.cᵍ * a, 0, zk, 0, merge(p, (; x₀ = 0.0))) for zk in z, a in ages]
Uᴸ = Uˢ .+ ΔU_mean

# Shear ratios in the upper Stokes layer (0 > z > −1/k₀), by age
∂z(F) = (F[2:end, :] .- F[1:end-1, :]) ./ diff(z)
top = findall(zz -> zz > -1 / k, zf[2:end-1])
R = -∂z(ΔU_mean) ./ ∂z(Uˢ)
Rᴸ = ∂z(Uᴸ) ./ ∂z(Uˢ)

hr("Lagrangian mean beneath the packet: case $(case.name), $level, seeds $seeds")
println("  age (τ₀)   uˢ(0)   ΔU(0)    uᴸ'(0)    ⟨R⟩ₜₒₚ = −∂zΔU/∂zuˢ   ⟨∂zuᴸ/∂zuˢ⟩ₜₒₚ    [mm/s, means over 0 > z > −1/k₀]")
for a0 in (-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2)
    ia = argmin(abs.(ages .- a0 * τ))
    @printf("   %5.2f   %6.2f   %6.2f   %6.2f        %6.2f              %6.2f\n", ages[ia] / τ, 1e3Uˢ[end, ia], 1e3ΔU_mean[end, ia],
            1e3Uᴸ[end, ia], mean(R[top, ia]), mean(Rᴸ[top, ia]))
end
@printf("  control anisotropy A = u'²/w'² over 0 > z > −1/k₀ during passage: %.2f (range %.2f–%.2f)\n",
        mean(A_mean[findall(zz -> zz > -1 / k, z)]), extrema(A_mean[findall(zz -> zz > -1 / k, z)])...)
ia0 = argmin(abs.(ages))
@printf("  at the packet centre the Eulerian response cancels %.0f %% of the Stokes shear; the Lagrangian mean retains %.0f %%\n",
        100mean(R[top, ia0]), 100mean(Rᴸ[top, ia0]))

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1800, 1000))
Label(fig[0, 1:3], "Case $(case.name), $level, $n seeds: Lagrangian mean beneath the packet (wake-age composites)", fontsize=22)

kz = k .* z
ax1 = Axis(fig[1, 1]; xlabel="mm/s", ylabel="k₀ z", title="(1) at the packet centre (age 0): uˢ, Eulerian ΔU, Lagrangian uˢ + ΔU")
lines!(ax1, 1e3 .* Uˢ[:, ia0], kz; color=:gray, linewidth=3, label="uˢ (prescribed)")
band!(ax1, Point2f.(1e3 .* (ΔU_mean[:, ia0] .- ΔU_err[:, ia0]), kz), Point2f.(1e3 .* (ΔU_mean[:, ia0] .+ ΔU_err[:, ia0]), kz); color=(:firebrick, 0.25))
lines!(ax1, 1e3 .* ΔU_mean[:, ia0], kz; color=:firebrick, linewidth=3, label="ΔU (Eulerian, null-corrected)")
lines!(ax1, 1e3 .* Uᴸ[:, ia0], kz; color=:royalblue, linewidth=3, label="uˢ + ΔU (Lagrangian-mean change)")
# Interior anisotropy (below the surface blocking layer) for the quasi-equilibrium reference
Ā = mean(A_mean[findall(zz -> -2 / k < zz < -0.5 / k, z)])
lines!(ax1, 1e3 .* (1 - Ā) .* Uˢ[:, ia0], kz; color=:black, linestyle=:dash,
       label=@sprintf("(1 − A) uˢ: full quasi-equilibrium, interior A = %.1f", Ā))
vlines!(ax1, [0]; color=(:black, 0.3))
ylims!(ax1, -4, 0)
xlims!(ax1, 1e3 * min(1.2 * (1 - Ā) * p.Uˢ₀, -0.02), 1e3 * 1.1 * p.Uˢ₀)
axislegend(ax1; position=:rb, labelsize=13)

ax2 = Axis(fig[1, 2]; xlabel="age (x_c − x)/cᵍ (τ₀)", ylabel="mm/s", title="(2) surface values vs age")
lines!(ax2, ages ./ τ, 1e3 .* Uˢ[end, :]; color=:gray, linewidth=3, label="uˢ")
lines!(ax2, ages ./ τ, 1e3 .* ΔU_mean[end, :]; color=:firebrick, linewidth=3, label="ΔU")
lines!(ax2, ages ./ τ, 1e3 .* Uᴸ[end, :]; color=:royalblue, linewidth=3, label="uˢ + ΔU")
vlines!(ax2, [0]; color=(:black, 0.3), linestyle=:dot)
hlines!(ax2, [0]; color=(:black, 0.3))
xlims!(ax2, -3, 4)
axislegend(ax2; position=:rt)

ax3 = Axis(fig[1, 3]; xlabel="age (τ₀)", ylabel="ratio", title="(3) shear ratios over 0 > z > −1/k₀")
lines!(ax3, ages ./ τ, [mean(R[top, ia]) for ia in eachindex(ages)]; color=:firebrick, linewidth=3, label="R = −∂zΔU / ∂zuˢ")
lines!(ax3, ages ./ τ, [mean(Rᴸ[top, ia]) for ia in eachindex(ages)]; color=:royalblue, linewidth=3, label="∂zuᴸ / ∂zuˢ = 1 − R")
hlines!(ax3, [1, Ā]; color=:black, linestyle=:dash)
text!(ax3, -2.9, Ā + 0.05; text="interior A = u'²/w'² (full quasi-equilibrium)", fontsize=13)
text!(ax3, -2.9, 1.05; text="1 (homogenized Lagrangian mean)", fontsize=13)
hlines!(ax3, [0]; color=(:black, 0.3))
vlines!(ax3, [0]; color=(:black, 0.3), linestyle=:dot)
xlims!(ax3, -3, 3)
ylims!(ax3, -0.5, 3)
axislegend(ax3; position=:rt)

output = get(args, "output", joinpath(figure_directory(), "lagrangian_mean_$(case_dirname(case))_$(level)$(isempty(topology_tag(x_topology)) ? "" : "_" * topology_tag(x_topology)).png"))
save(output, fig)
@info "Saved $output"
