#####
##### Reproduction of Ellingsen et al. (2026) figure 3(a) for case 1.D with the paper's
##### averaging scheme.
#####
##### The SPIV plane is fixed in the laboratory; the paper averages the streamwise velocity over
##### the field of view, over the ensemble of 60 wave groups, and over the time intervals of
##### figure 2(b): interval 1 (before the group) is −4.5τ < t − tₚ < −3.3τ and interval 3 (after
##### the group) is 2.0τ < t − tₚ < 3.2τ, with τ the laboratory group width and tₚ the time of the
##### group peak at the plane. A fluid parcel observed at the plane at lab time t − tₚ was passed by
##### the group peak (c_g − U₀)/c_g (t − tₚ) = (t − tₚ) τ₀/τ earlier, so in the current-following
##### frame the same intervals are −4.5τ₀ < age < −3.3τ₀ and 2.0τ₀ < age < 3.2τ₀ at a plane fixed in
##### the fluid, which is what the bounded-tank runs provide directly (t_peak = 22.9 s at x_FOV).
##### U₁(z) and U₃(z) are averaged over y (0.8 m; the SPIV FOV was 0.12 m wide) and the interval,
##### then over seeds. As in the paper, ΔU = U₃ − U₁ uses the wave member alone (no control or
##### null subtraction); the paired, null-corrected estimate is shown for reference.
#####
##### Sampling: the laboratory plane is fixed while the turbulence advects through it at U₀, so
##### each of the 60 groups contributes ~U₀ × 1.2τ ≈ 3 m of independent fluid per interval. A plane
##### fixed in the fluid frame sees the same eddies evolve for 1.2τ₀ ≈ 3 s (well under an eddy
##### turnover), i.e. a handful of samples per seed, and the raw U₃ − U₁ is then dominated by
##### turbulence noise (±5 mm/s with four seeds). The paper's scheme is therefore also applied to
##### every fluid column of the tank at the same ages (the wake-age composite): identical
##### definitions of U₁ and U₃, ~50 integral scales per seed instead of one.
#####
##### Usage: sbatch batch/anti_stokes_analysis.batch script=figure3_reproduction.jl case=1.D level=M2 seeds=1,2,3,4 x_topology=bounded
#####

using CairoMakie
using DelimitedFiles
include("common.jl")

args = parse_key_value_args(ARGS)
case_name = getarg(args, "case", "1.D")
level = getarg(args, "level", "M2")
seeds = parse.(Int, split(getarg(args, "seeds", "1,2,3,4"), ','))
numerics = getarg(args, "numerics", "weno")
Δt = getarg(args, "dt", 0.02)
x_topology = getarg(args, "x_topology", "bounded")
root = getarg(args, "root", default_data_root())
interval_1 = (-4.5, -3.3)     # figure 2(b), in units of τ
interval_3 = (2.0, 3.2)

case = anti_stokes_case(case_name)
U₀ = Float64(case.U₀)

# Digitized experimental curves (analysis/anti_stokes/data/, extracted from the paper's vector graphics)
datadir = joinpath(@__DIR__, "data")
exp3a = readdlm(joinpath(datadir, "ellingsen2026_fig3a_1D.csv"), ',', Float64; comments=true)
exp3b_raw = readdlm(joinpath(datadir, "ellingsen2026_fig3b.csv"), ',', String; comments=true)
exp3b = [(strip(r[1]), parse(Float64, r[2]), parse(Float64, r[3])) for r in eachrow(exp3b_raw) if strip(r[1]) == case_name]

interval_mean(A, t, tp, τ, (a, b)) = window_mean(A, t, tp + a * τ, tp + b * τ)
# the same interval applied to every fluid column: mean over all (x, t) with age in the interval
function age_mean(A, x, t, p, τ, (a, b); keep=Colon())
    _, C, N = wake_age_composite(A[keep, :, :], x[keep], t, p; age_edges=[a * τ, b * τ])
    N[1] > 0 || error("no samples with age in [$a, $b] τ₀")
    return C[:, 1], N[1]
end

U1s, U3s, ΔUpaired = [], [], []
U1c, U3c, ΔUpc = [], [], []
z, k = nothing, nothing
null_dir = run_directory(root, case, level, "packet_null"; seed=0, Δt, numerics, x_topology)
quiescent_dir = run_directory(root, case, level, "quiescent_control"; seed=0, Δt, numerics, x_topology)
isdir(quiescent_dir) || (quiescent_dir = nothing)
for seed in seeds
    pk_dir = run_directory(root, case, level, "packet_turbulence"; seed, Δt, numerics, x_topology)
    ct_dir = run_directory(root, case, level, "turbulence_control"; seed, Δt, numerics, x_topology)
    isdir(pk_dir) || (@warn "missing $pk_dir"; continue)
    pk = load_run(pk_dir; fields=("U",))
    t, i, τ, tp = times(pk), fov_index(pk), τ₀(pk), t_peak(pk)
    global z, k = znodes_centers(pk), k₀(pk)
    tp + interval_1[1] * τ >= t[1] - 1e-6 || @warn "interval 1 starts before the run ($(tp + interval_1[1]τ) s)"
    tp + interval_3[2] * τ <= t[end] + 1e-6 || @warn "interval 3 ends after the run"
    UE = eulerian_U(pk)
    plane_ok = tp + interval_1[1] * τ >= t[1] - 1e-6 && tp + interval_3[2] * τ <= t[end] + 1e-6
    nanprof = fill(NaN, length(z))
    push!(U1s, plane_ok ? interval_mean(UE, t, tp, τ, interval_1)[i, :] : nanprof)
    push!(U3s, plane_ok ? interval_mean(UE, t, tp, τ, interval_3)[i, :] : nanprof)
    # paired, null-corrected reference
    ΔU, _, _ = paired_residual(pk_dir, ct_dir, null_dir, quiescent_dir; fields=("U",))
    push!(ΔUpaired, plane_ok ? interval_mean(ΔU, t, tp, τ, interval_3)[i, :] .- interval_mean(ΔU, t, tp, τ, interval_1)[i, :] : nanprof)
    # the same intervals applied to every fluid column (excluding 1 m next to the end walls)
    x, p = xnodes_faces(pk), run_packet(pk)
    keep = is_bounded_x(pk) ? findall(xi -> 1.0 <= xi <= pk.meta["Lx"] - 1.0, x) : Colon()
    c1, n1 = age_mean(UE, x, t, p, τ, interval_1; keep)
    c3, n3 = age_mean(UE, x, t, p, τ, interval_3; keep)
    push!(U1c, c1); push!(U3c, c3)
    push!(ΔUpc, age_mean(ΔU, x, t, p, τ, interval_3; keep)[1] .- age_mean(ΔU, x, t, p, τ, interval_1; keep)[1])
    @info @sprintf("seed %d: interval 1 = [%.1f, %.1f] s, interval 3 = [%.1f, %.1f] s at the plane: raw surface ΔU = %.2f mm/s, paired %.2f mm/s; all columns (%d/%d column-times): raw %.2f, paired %.2f mm/s",
                   seed, tp + interval_1[1]τ, tp + interval_1[2]τ, tp + interval_3[1]τ, tp + interval_3[2]τ,
                   1e3 * (U3s[end][end] - U1s[end][end]), 1e3 * ΔUpaired[end][end], n1, n3,
                   1e3 * (c3[end] - c1[end]), 1e3 * ΔUpc[end][end])
end
n = length(U1s)
n > 0 || error("no runs found")
U1 = hcat(U1s...); U3 = hcat(U3s...); P = hcat(ΔUpaired...)
nanmean(A) = [(v = filter(!isnan, A[j, :]); isempty(v) ? NaN : mean(v)) for j in 1:size(A, 1)]
nanse(A) = [(v = filter(!isnan, A[j, :]); length(v) < 2 ? NaN : std(v) / sqrt(length(v))) for j in 1:size(A, 1)]
U1m, U3m = nanmean(U1), nanmean(U3)
ΔUraw = U3 .- U1
ΔUm, ΔUse = nanmean(ΔUraw), nanse(ΔUraw)
Pm, Pse = nanmean(P), nanse(P)
U1C, U3C, PC = hcat(U1c...), hcat(U3c...), hcat(ΔUpc...)
U1Cm, U3Cm = vec(mean(U1C; dims=2)), vec(mean(U3C; dims=2))
ΔUC = U3C .- U1C
ΔUCm, ΔUCse = vec(mean(ΔUC; dims=2)), vec(std(ΔUC; dims=2)) ./ sqrt(n)
PCm, PCse = vec(mean(PC; dims=2)), vec(std(PC; dims=2)) ./ sqrt(n)
kz = k .* z

# Surface value and depth of the paper's window (k₀z > −1.2)
shallow = kz .>= -1.2
@info @sprintf("Simulation (%d seeds), fixed plane: surface ΔU raw = %.2f ± %.2f mm/s, paired = %.2f ± %.2f mm/s", n, 1e3ΔUm[end], 1e3ΔUse[end], 1e3Pm[end], 1e3Pse[end])
@info @sprintf("Simulation (%d seeds), all columns: surface ΔU raw = %.2f ± %.2f mm/s, paired = %.2f ± %.2f mm/s; experiment (digitized) ΔU at k₀z = −0.1: %.2f mm/s",
               n, 1e3ΔUCm[end], 1e3ΔUCse[end], 1e3PCm[end], 1e3PCse[end], exp3b[end][3])

# shape comparison: remove the mean over the PIV window k₀z ∈ [−1.2, −0.1] from both
win = findall(kk -> -1.2 <= kk <= -0.1, kz)
exp_kz = [r[2] for r in exp3b]; exp_dU = [r[3] for r in exp3b]
exp_win = findall(kk -> -1.2 <= kk <= -0.1, exp_kz)
exp_mean = mean(exp_dU[exp_win]); les_mean = 1e3 * mean(ΔUCm[win]); les_pmean = 1e3 * mean(PCm[win])
zero_crossing(v, kzv) = (j = findfirst(x -> x > 0, v); isnothing(j) || j == 1 ? NaN : kzv[j-1] + (kzv[j] - kzv[j-1]) * (0 - v[j-1]) / (v[j] - v[j-1]))
@info @sprintf("Window means (k₀z ∈ [−1.2, −0.1]): experiment %.2f mm/s, LES raw %.2f, LES paired %.2f → offset exp − LES = %.2f mm/s", exp_mean, les_mean, les_pmean, exp_mean - les_mean)
@info @sprintf("Zero crossing of ΔU: experiment k₀z = %.2f, LES raw %.2f, LES paired %.2f", zero_crossing(reverse(exp_dU), reverse(exp_kz)), zero_crossing(1e3ΔUCm, kz), zero_crossing(1e3PCm, kz))

set_theme!(Theme(fontsize=18))
fig = Figure(size=(2400, 620))
Label(fig[0, 1:4], "Ellingsen et al. (2026) figure 3(a), case $case_name: U₁ (interval 1, −4.5τ to −3.3τ before the peak) and U₃ (interval 3, 2.0τ to 3.2τ after), " *
                   "experiment vs LES ($level, $(is_bounded_x(load_run(run_directory(root, case, level, "packet_null"; seed=0, Δt, numerics, x_topology); fields=("U",))) ? "bounded tank" : "periodic"), $n seeds)", fontsize=20)

ax1 = Axis(fig[1, 1]; xlabel="U (m/s)", ylabel="k₀ z", title="(a) experiment (digitized from the paper)")
lines!(ax1, exp3a[:, 2], exp3a[:, 1]; color=:dodgerblue, linewidth=3, label="U₁")
lines!(ax1, exp3a[:, 3], exp3a[:, 1]; color=:orangered, linewidth=3, linestyle=:dashdot, label="U₃")
xlims!(ax1, -0.35, -0.33); ylims!(ax1, -1.2, 0)
axislegend(ax1; position=:lb)

ax2 = Axis(fig[1, 2]; xlabel="U (m/s), laboratory frame: −U₀ + ⟨u⟩", ylabel="k₀ z", title="(b) LES: same intervals, every fluid column (thin: one plane, per seed)")
lines!(ax2, -U₀ .+ U1Cm, kz; color=:dodgerblue, linewidth=3, label="U₁")
lines!(ax2, -U₀ .+ U3Cm, kz; color=:orangered, linewidth=3, linestyle=:dashdot, label="U₃")
for s in 1:n
    lines!(ax2, -U₀ .+ U1[:, s], kz; color=(:dodgerblue, 0.2), linewidth=1)
    lines!(ax2, -U₀ .+ U3[:, s], kz; color=(:orangered, 0.2), linewidth=1)
end
xlims!(ax2, -U₀ - 0.015, -U₀ + 0.005); ylims!(ax2, -1.2, 0)
axislegend(ax2; position=:lb)

ax3 = Axis(fig[1, 3]; xlabel="ΔU = U₃ − U₁ (mm/s)", ylabel="k₀ z", title="(c) ΔU = U₃ − U₁: experiment vs LES")
lines!(ax3, [r[3] for r in exp3b], [r[2] for r in exp3b]; color=:darkred, linewidth=3, label="experiment $case_name (figure 3b)")
band!(ax3, Point2f.(1e3 .* (ΔUCm .- ΔUCse), kz), Point2f.(1e3 .* (ΔUCm .+ ΔUCse), kz); color=(:black, 0.15))
lines!(ax3, 1e3 .* ΔUCm, kz; color=:black, linewidth=3, label="LES raw U₃ − U₁, all columns (± s.e., $n seeds)")
lines!(ax3, 1e3 .* PCm, kz; color=:gray40, linewidth=2, linestyle=:dash, label="LES paired, null-corrected, all columns")
lines!(ax3, 1e3 .* ΔUm, kz; color=(:steelblue, 0.8), linewidth=1.5, label=@sprintf("LES raw at one plane (s.e. %.1f mm/s at the surface)", 1e3ΔUse[end]))
uˢ_surface = Float64(case.Uˢ₀)
lines!(ax3, -1e3 .* uˢ_surface .* exp.(2kz), kz; color=(:gray, 0.6), linestyle=:dot, label="−uˢ(z) at the group peak")
vlines!(ax3, [0]; color=(:black, 0.3))
xlims!(ax3, -15, 5); ylims!(ax3, -1.2, 0)
axislegend(ax3; position=:lb, labelsize=13)

ax4 = Axis(fig[1, 4]; xlabel="ΔU − ⟨ΔU⟩_window (mm/s)", ylabel="k₀ z", title="(d) shape: window mean removed from both")
lines!(ax4, exp_dU .- exp_mean, exp_kz; color=:darkred, linewidth=3, label="experiment")
lines!(ax4, 1e3 .* ΔUCm .- les_mean, kz; color=:black, linewidth=3, label="LES raw, all columns")
lines!(ax4, 1e3 .* PCm .- les_pmean, kz; color=:gray40, linewidth=2, linestyle=:dash, label="LES paired")
vlines!(ax4, [0]; color=(:black, 0.3)); xlims!(ax4, -10, 6); ylims!(ax4, -1.2, 0); axislegend(ax4; position=:lb, labelsize=13)

output = get(args, "output", joinpath(figure_directory(), "figure3a_reproduction_" * replace(case_name, "." => "") * "_$(level)_$(x_topology).png"))
save(output, fig)
@info "Saved $output"

# Write the LES profiles alongside the digitized data for reuse
open(joinpath(figure_directory(), "figure3a_reproduction_" * replace(case_name, "." => "") * "_$(level)_$(x_topology).csv"), "w") do io
    println(io, "# all fluid columns: k0z, U1_lab [m/s], U3_lab [m/s], dU_raw [mm/s], dU_raw_se [mm/s], dU_paired [mm/s], dU_paired_se [mm/s], dU_raw_one_plane [mm/s], dU_raw_one_plane_se [mm/s]  ($n seeds)")
    for j in eachindex(kz)
        @printf(io, "%.5f, %.6f, %.6f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n", kz[j], -U₀ + U1Cm[j], -U₀ + U3Cm[j], 1e3ΔUCm[j], 1e3ΔUCse[j], 1e3PCm[j], 1e3PCse[j], 1e3ΔUm[j], 1e3ΔUse[j])
    end
end
