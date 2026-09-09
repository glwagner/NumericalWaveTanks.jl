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

U1s, U3s, ΔUpaired = [], [], []
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
    push!(U1s, interval_mean(UE, t, tp, τ, interval_1)[i, :])
    push!(U3s, interval_mean(UE, t, tp, τ, interval_3)[i, :])
    # paired, null-corrected reference
    ΔU, _, _ = paired_residual(pk_dir, ct_dir, null_dir, quiescent_dir)
    push!(ΔUpaired, interval_mean(ΔU, t, tp, τ, interval_3)[i, :] .- interval_mean(ΔU, t, tp, τ, interval_1)[i, :])
    @info @sprintf("seed %d: interval 1 = [%.1f, %.1f] s, interval 3 = [%.1f, %.1f] s, raw surface ΔU = %.2f mm/s, paired %.2f mm/s",
                   seed, tp + interval_1[1]τ, tp + interval_1[2]τ, tp + interval_3[1]τ, tp + interval_3[2]τ,
                   1e3 * (U3s[end][end] - U1s[end][end]), 1e3 * ΔUpaired[end][end])
end
n = length(U1s)
n > 0 || error("no runs found")
U1 = hcat(U1s...); U3 = hcat(U3s...); P = hcat(ΔUpaired...)
U1m, U3m = vec(mean(U1; dims=2)), vec(mean(U3; dims=2))
ΔUraw = U3 .- U1
ΔUm, ΔUse = vec(mean(ΔUraw; dims=2)), vec(std(ΔUraw; dims=2)) ./ sqrt(n)
Pm, Pse = vec(mean(P; dims=2)), vec(std(P; dims=2)) ./ sqrt(n)
kz = k .* z

# Surface value and depth of the paper's window (k₀z > −1.2)
shallow = kz .>= -1.2
@info @sprintf("Simulation (%d seeds): surface ΔU raw = %.2f ± %.2f mm/s, paired = %.2f ± %.2f mm/s; experiment (digitized) ΔU at k₀z = −0.1: %.2f mm/s",
               n, 1e3ΔUm[end], 1e3ΔUse[end], 1e3Pm[end], 1e3Pse[end], exp3b[end][3])

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1800, 620))
Label(fig[0, 1:3], "Ellingsen et al. (2026) figure 3(a), case $case_name: U₁ (interval 1, −4.5τ to −3.3τ before the peak) and U₃ (interval 3, 2.0τ to 3.2τ after), " *
                   "experiment vs LES ($level, $(is_bounded_x(load_run(run_directory(root, case, level, "packet_null"; seed=0, Δt, numerics, x_topology); fields=("U",))) ? "bounded tank" : "periodic"), $n seeds)", fontsize=20)

ax1 = Axis(fig[1, 1]; xlabel="U (m/s)", ylabel="k₀ z", title="(a) experiment (digitized from the paper)")
lines!(ax1, exp3a[:, 2], exp3a[:, 1]; color=:dodgerblue, linewidth=3, label="U₁")
lines!(ax1, exp3a[:, 3], exp3a[:, 1]; color=:orangered, linewidth=3, linestyle=:dashdot, label="U₃")
xlims!(ax1, -0.35, -0.33); ylims!(ax1, -1.2, 0)
axislegend(ax1; position=:lb)

ax2 = Axis(fig[1, 2]; xlabel="U (m/s), laboratory frame: −U₀ + ⟨u⟩", ylabel="k₀ z", title="(b) LES, same intervals, y-averaged, mean over seeds")
lines!(ax2, -U₀ .+ U1m, kz; color=:dodgerblue, linewidth=3, label="U₁")
lines!(ax2, -U₀ .+ U3m, kz; color=:orangered, linewidth=3, linestyle=:dashdot, label="U₃")
for s in 1:n
    lines!(ax2, -U₀ .+ U1[:, s], kz; color=(:dodgerblue, 0.25), linewidth=1)
    lines!(ax2, -U₀ .+ U3[:, s], kz; color=(:orangered, 0.25), linewidth=1)
end
xlims!(ax2, -U₀ - 0.015, -U₀ + 0.005); ylims!(ax2, -1.2, 0)
axislegend(ax2; position=:lb)

ax3 = Axis(fig[1, 3]; xlabel="ΔU = U₃ − U₁ (mm/s)", ylabel="k₀ z", title="(c) ΔU: experiment vs LES (raw, as in the paper; paired null-corrected for reference)")
lines!(ax3, [r[3] for r in exp3b], [r[2] for r in exp3b]; color=:darkred, linewidth=3, label="experiment $case_name (figure 3b)")
band!(ax3, Point2f.(1e3 .* (ΔUm .- ΔUse), kz), Point2f.(1e3 .* (ΔUm .+ ΔUse), kz); color=(:black, 0.15))
lines!(ax3, 1e3 .* ΔUm, kz; color=:black, linewidth=3, label="LES raw U₃ − U₁ (± s.e., $n seeds)")
lines!(ax3, 1e3 .* Pm, kz; color=:gray40, linewidth=2, linestyle=:dash, label="LES paired, null-corrected")
uˢ_surface = Float64(case.Uˢ₀)
lines!(ax3, -1e3 .* uˢ_surface .* exp.(2kz), kz; color=(:gray, 0.6), linestyle=:dot, label="−uˢ(z) at the group peak")
vlines!(ax3, [0]; color=(:black, 0.3))
xlims!(ax3, -15, 5); ylims!(ax3, -1.2, 0)
axislegend(ax3; position=:lb, labelsize=13)

output = get(args, "output", joinpath(figure_directory(), "figure3a_reproduction_" * replace(case_name, "." => "") * "_$(level)_$(x_topology).png"))
save(output, fig)
@info "Saved $output"

# Write the LES profiles alongside the digitized data for reuse
open(joinpath(figure_directory(), "figure3a_reproduction_" * replace(case_name, "." => "") * "_$(level)_$(x_topology).csv"), "w") do io
    println(io, "# k0z, U1_lab [m/s], U3_lab [m/s], dU_raw [mm/s], dU_raw_se [mm/s], dU_paired [mm/s], dU_paired_se [mm/s]  ($n seeds)")
    for j in eachindex(kz)
        @printf(io, "%.5f, %.6f, %.6f, %.4f, %.4f, %.4f, %.4f\n", kz[j], -U₀ + U1m[j], -U₀ + U3m[j], 1e3ΔUm[j], 1e3ΔUse[j], 1e3Pm[j], 1e3Pse[j])
    end
end
