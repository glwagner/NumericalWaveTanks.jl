#####
##### Horizontally uniform wave group (temporal envelope) and the Langmuir-shear variant, compared
##### with the travelling packet, case 1.D by default.
#####
#####   uniform_packet_turbulence  — uˢ(z, t) = Uˢ₀ exp(−((t − t_peak)/τ₀)²) e^{2kz}, same checkpoints,
#####                                duration and control (turbulence_control) as the travelling packet
#####   sheared_control            — turbulence + initial Eulerian current α Uˢ₀ e^{2kz}, no waves
#####   sheared_packet_turbulence  — the same current under the uniform group (CL2-unstable: the
#####                                Eulerian and Stokes shears are aligned, Langmuir circulation)
#####   uniform_packet_null / sheared_packet_null — the quiescent references
#####
##### Because the forcing is uniform in x, every profile is averaged over x as well as y, which
##### gives ~50 integral scales per seed; the travelling-packet comparison uses the wake-age
##### composite of the same seeds (age = time since the group peak passed a fluid column).
#####
##### Usage: sbatch batch/anti_stokes_analysis.batch script=uniform_packet.jl case=1.D level=M2 seeds=1,2,3,4
#####

using CairoMakie
using FFTW
include("common.jl")

args = parse_key_value_args(ARGS)
case_name = getarg(args, "case", "1.D")
level = getarg(args, "level", "M2")
seeds = parse.(Int, split(getarg(args, "seeds", "1,2,3,4"), ','))
numerics = getarg(args, "numerics", "weno")
Δt = getarg(args, "dt", 0.02)
root = getarg(args, "root", default_data_root())
case = anti_stokes_case(case_name)
Uˢ₀ = Float64(case.Uˢ₀)

dir(member; seed=0) = run_directory(root, case, level, member; seed, Δt, numerics)
xmean(A) = dropdims(mean(A; dims=1); dims=1)          # (Nx, Nz, Nt) → (Nz, Nt)

quiescent = dir("quiescent_control")
isdir(quiescent) || (quiescent = nothing)

ages_profiles = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
interval_3 = (2.0, 3.2)                                 # the paper's interval 3

res = (; ΔU_uniform = [], ΔU_travel = [], surf_uniform = [], surf_travel = [], U_sh = [], U_sc = [], U_sn = [],
         ΔU_sh = [], wrms = Dict(m => [] for m in ("uniform_packet_turbulence", "turbulence_control", "sheared_packet_turbulence", "sheared_control")),
         spectra = Dict(m => [] for m in ("uniform_packet_turbulence", "sheared_packet_turbulence", "sheared_control")))
z = zf = t = kz = ages = nothing; τ = tp = k = 0.0
for seed in seeds
    un_dir, ct_dir, pk_dir = dir("uniform_packet_turbulence"; seed), dir("turbulence_control"; seed), dir("packet_turbulence"; seed)
    sh_dir, sc_dir = dir("sheared_packet_turbulence"; seed), dir("sheared_control"; seed)
    all(isdir, (un_dir, ct_dir, sh_dir, sc_dir)) || (@warn "seed $seed incomplete"; continue)
    un = load_run(un_dir; fields=("U", "W", "WW"))
    global t, z, zf, k = times(un), znodes_centers(un), znodes_faces(un), k₀(un)
    global τ, tp = τ₀(un), t_peak(un)
    global kz = k .* z
    global ages = (t .- tp) ./ τ
    # uniform group: paired, null-corrected Eulerian change, x-averaged
    ΔU, _, _ = paired_residual(un_dir, ct_dir, dir("uniform_packet_null"), quiescent; fields=("U",))
    ΔUz = xmean(ΔU)
    push!(res.ΔU_uniform, ΔUz)
    push!(res.surf_uniform, ΔUz[end, :])
    # travelling packet, same seed: wake-age composite
    if isdir(pk_dir)
        ΔUp, pk, _ = paired_residual(pk_dir, ct_dir, dir("packet_null"), quiescent; fields=("U",))
        edges = collect(range(-4.5τ, 4.5τ; step=τ / 8))
        a, C, _ = wake_age_composite(ΔUp, xnodes_faces(pk), times(pk), run_packet(pk); age_edges=edges)
        push!(res.ΔU_travel, (a ./ τ, C))
        push!(res.surf_travel, (a ./ τ, C[end, :]))
    end
    # sheared members: x-averaged Eulerian mean current
    sh = load_run(sh_dir; fields=("U", "W", "WW"))
    sc = load_run(sc_dir; fields=("U", "W", "WW"))
    push!(res.U_sh, xmean(eulerian_U(sh)))
    push!(res.U_sc, xmean(xzt(sc, "U")))
    sn_dir = dir("sheared_packet_null")
    isdir(sn_dir) && push!(res.U_sn, xmean(eulerian_U(load_run(sn_dir; fields=("U",)))))
    push!(res.ΔU_sh, res.U_sh[end] .- res.U_sc[end])
    # vertical-velocity variance profiles (x-averaged central moments)
    for (m, r) in (("uniform_packet_turbulence", un), ("turbulence_control", load_run(ct_dir; fields=("W", "WW"))),
                   ("sheared_packet_turbulence", sh), ("sheared_control", sc))
        W, WW = xzt(r, "W"), xzt(r, "WW")
        push!(res.wrms[m], sqrt.(max.(xmean(WW .- W .^ 2), 0)))
    end
    # spanwise spectra of w at z ≈ −δˢ from the y-z plane, averaged over 0 < age < 2τ₀
    kδ = argmin(abs.(zf .+ 1 / (2k)))
    for (m, d) in (("uniform_packet_turbulence", un_dir), ("sheared_packet_turbulence", sh_dir), ("sheared_control", sc_dir))
        wts = FieldTimeSeries(joinpath(d, "fov_plane.jld2"), "w")
        tt = collect(Float64, wts.times)
        sel = findall(τt -> tp <= τt <= tp + 2τ, tt)
        Ny = size(interior(wts[1]), 2)
        S = zeros(Ny ÷ 2 + 1)
        for n in sel
            wy = Float64.(vec(Array(interior(wts[n]))[1, :, kδ]))
            F = rfft(wy .- mean(wy))
            S .+= abs2.(F) ./ length(sel)
        end
        Ly = un.meta["Ly"]
        push!(res.spectra[m], (2π / Ly .* (0:Ny÷2), S))
    end
    @info @sprintf("seed %d: uniform surface ΔU at age 2–3.2τ₀ = %.2f mm/s; sheared surface U at peak %.2f (control %.2f) mm/s",
                   seed, 1e3 * mean(ΔUz[end, findall(a -> interval_3[1] <= a <= interval_3[2], ages)]),
                   1e3 * res.U_sh[end][end, nearest_index(t, tp)], 1e3 * res.U_sc[end][end, nearest_index(t, tp)])
end
n = length(res.ΔU_uniform)
n > 0 || error("no complete seeds")

sem(v) = std(v) / sqrt(length(v))
window(a₀, a₁) = findall(a -> a₀ <= a <= a₁, ages)
prof_uniform = [mean(ΔU[:, window(interval_3...)]; dims=2) |> vec for ΔU in res.ΔU_uniform]
Pu = hcat(prof_uniform...)
prof_travel = [composite_profile(a .* τ, C, τ, interval_3...) for (a, C) in res.ΔU_travel]
Pt = isempty(prof_travel) ? nothing : hcat(prof_travel...)
@info @sprintf("Surface ΔU over the paper's interval 3: uniform group %.2f ± %.2f mm/s%s (%d seeds); Uˢ₀ = %.1f mm/s",
               1e3 * mean(Pu[end, :]), 1e3 * sem(Pu[end, :]),
               isnothing(Pt) ? "" : @sprintf(", travelling packet %.2f ± %.2f mm/s", 1e3 * mean(Pt[end, :]), 1e3 * sem(Pt[end, :])), n, 1e3Uˢ₀)

set_theme!(Theme(fontsize=17))
fig = Figure(size=(2000, 1300))
Label(fig[0, 1:3], "Case $case_name at $level: horizontally uniform group vs travelling packet, and the Langmuir-shear variant (α = 1: initial Eulerian current Uˢ₀e^{2kz}), $n seeds", fontsize=21)
colors = Makie.wong_colors()

ax1 = Axis(fig[1, 1]; xlabel="ΔU (mm/s)", ylabel="k₀ z", title="(1) null-corrected ΔU(z) over ages 2–3.2τ₀ (paper's interval 3)")
band!(ax1, Point2f.(1e3 .* (vec(mean(Pu; dims=2)) .- vec(std(Pu; dims=2)) ./ sqrt(n)), kz), Point2f.(1e3 .* (vec(mean(Pu; dims=2)) .+ vec(std(Pu; dims=2)) ./ sqrt(n)), kz); color=(colors[1], 0.2))
lines!(ax1, 1e3 .* vec(mean(Pu; dims=2)), kz; color=colors[1], linewidth=3, label="uniform group")
if !isnothing(Pt)
    band!(ax1, Point2f.(1e3 .* (vec(mean(Pt; dims=2)) .- vec(std(Pt; dims=2)) ./ sqrt(n)), kz), Point2f.(1e3 .* (vec(mean(Pt; dims=2)) .+ vec(std(Pt; dims=2)) ./ sqrt(n)), kz); color=(colors[2], 0.2))
    lines!(ax1, 1e3 .* vec(mean(Pt; dims=2)), kz; color=colors[2], linewidth=3, label="travelling packet (wake-age composite)")
end
lines!(ax1, -1e3 .* Uˢ₀ .* exp.(2kz), kz; color=(:gray, 0.6), linestyle=:dash, label="−uˢ at the peak")
vlines!(ax1, [0]; color=(:black, 0.3)); ylims!(ax1, -4, 0); axislegend(ax1; position=:lb, labelsize=13)

ax2 = Axis(fig[1, 2]; xlabel="age (t − t_peak)/τ₀", ylabel="surface ΔU (mm/s)", title="(2) surface response against group age")
for (s, su) in enumerate(res.surf_uniform)
    lines!(ax2, ages, 1e3 .* su; color=(colors[1], 0.3), linewidth=1)
end
Su = hcat(res.surf_uniform...)
lines!(ax2, ages, 1e3 .* vec(mean(Su; dims=2)); color=colors[1], linewidth=3, label="uniform group (seed mean)")
if !isempty(res.surf_travel)
    a₁ = res.surf_travel[1][1]
    St = hcat([s[2] for s in res.surf_travel]...)
    lines!(ax2, a₁, 1e3 .* vec(mean(St; dims=2)); color=colors[2], linewidth=3, label="travelling packet composite")
end
lines!(ax2, ages, -1e3 .* Uˢ₀ .* exp.(-ages .^ 2); color=(:gray, 0.6), linestyle=:dash, label="−uˢ(0, t)")
vspan!(ax2, interval_3...; color=(:orange, 0.12))
hlines!(ax2, [0]; color=(:black, 0.3)); xlims!(ax2, -4.5, 4.5); axislegend(ax2; position=:lb, labelsize=13)

ax3 = Axis(fig[1, 3]; xlabel="⟨u⟩ᴱ (mm/s)", ylabel="k₀ z", title="(3) sheared case: Eulerian mean current at ages −4…3τ₀ (solid: with group; dashed: sheared control)")
Ush = mean(cat(res.U_sh...; dims=3); dims=3)[:, :, 1]
Usc = mean(cat(res.U_sc...; dims=3); dims=3)[:, :, 1]
for (j, a) in enumerate(ages_profiles)
    nidx = nearest_index(ages, a)
    lines!(ax3, 1e3 .* Ush[:, nidx], kz; color=colors[mod1(j, 7)], linewidth=2.5, label=@sprintf("age %+.0fτ₀", a))
    lines!(ax3, 1e3 .* Usc[:, nidx], kz; color=colors[mod1(j, 7)], linewidth=1.5, linestyle=:dash)
end
lines!(ax3, 1e3 .* Uˢ₀ .* exp.(2kz), kz; color=(:black, 0.4), linestyle=:dot, label="initial current Uˢ₀e^{2kz}")
vlines!(ax3, [0]; color=(:black, 0.3)); ylims!(ax3, -4, 0); axislegend(ax3; position=:rb, labelsize=12)

ax4 = Axis(fig[2, 1]; xlabel="ΔU = ⟨u⟩ᴱ_sheared+group − ⟨u⟩ᴱ_sheared control (mm/s)", ylabel="k₀ z", title="(4) group-induced change of the sheared current vs the unsheared response")
ΔUsh = mean(cat(res.ΔU_sh...; dims=3); dims=3)[:, :, 1]
ΔUun = mean(cat(res.ΔU_uniform...; dims=3); dims=3)[:, :, 1]
for (j, a) in enumerate((0.0, 1.0, 2.0, 3.0))
    nidx = nearest_index(ages, a)
    lines!(ax4, 1e3 .* ΔUsh[:, nidx], kz; color=colors[j], linewidth=3, label=@sprintf("sheared, age %+.0fτ₀", a))
    lines!(ax4, 1e3 .* ΔUun[:, nidx], kz; color=colors[j], linewidth=1.5, linestyle=:dash, label=@sprintf("unsheared, age %+.0fτ₀", a))
end
vlines!(ax4, [0]; color=(:black, 0.3)); ylims!(ax4, -4, 0); axislegend(ax4; position=:lb, labelsize=12)

ax5 = Axis(fig[2, 2]; xlabel="w_rms (mm/s)", ylabel="k₀ z", title="(5) vertical-velocity rms at age 0 (solid) and 2τ₀ (dashed)")
for (j, m) in enumerate(("turbulence_control", "uniform_packet_turbulence", "sheared_control", "sheared_packet_turbulence"))
    Wm = mean(cat(res.wrms[m]...; dims=3); dims=3)[:, :, 1]
    kzw = k .* zf[1:size(Wm, 1)]                     # w statistics live on z faces
    lines!(ax5, 1e3 .* Wm[:, nearest_index(ages, 0.0)], kzw; color=colors[j], linewidth=3, label=m)
    lines!(ax5, 1e3 .* Wm[:, nearest_index(ages, 2.0)], kzw; color=colors[j], linewidth=2, linestyle=:dash)
end
ylims!(ax5, -4, 0); axislegend(ax5; position=:rb, labelsize=12)

ax6 = Axis(fig[2, 3]; xlabel="spanwise wavenumber k_y (m⁻¹)", ylabel="spectrum of w at z = −δˢ (arbitrary)", title="(6) spanwise w spectra over 0 < age < 2τ₀ (Langmuir cells → peak at the roll scale)", xscale=log10, yscale=log10)
for (j, m) in enumerate(("uniform_packet_turbulence", "sheared_control", "sheared_packet_turbulence"))
    Sm = mean(hcat([s[2] for s in res.spectra[m]]...); dims=2) |> vec
    ky = res.spectra[m][1][1]
    lines!(ax6, ky[2:end], Sm[2:end] .+ 1e-30; color=colors[j+1], linewidth=3, label=m)
end
vlines!(ax6, [2k]; color=(:black, 0.4), linestyle=:dot, label="2k₀ (Stokes depth)")
axislegend(ax6; position=:lb, labelsize=12)

output = get(args, "output", joinpath(figure_directory(), "uniform_packet_" * replace(case_name, "." => "") * "_$(level)_$(numerics).png"))
save(output, fig)
@info "Saved $output"
