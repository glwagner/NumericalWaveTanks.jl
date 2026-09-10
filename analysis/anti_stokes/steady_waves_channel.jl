#####
##### Steady wave train in the 1.D packet tank (uˢ = Uˢ₀ e^{2kz} switched on at t = 0), without and
##### with the initial Eulerian current α Uˢ₀ e^{2kz} (Langmuir-unstable). Members:
#####   steady_waves_turbulence  vs turbulence_control  (null: steady_waves_null)
#####   steady_sheared_turbulence vs sheared_control    (laminar reference: steady_sheared_null with noise)
##### Everything is x- and y-averaged; time since onset replaces the group age.
#####
##### Usage: sbatch batch/anti_stokes_analysis.batch script=steady_waves_channel.jl case=1.D level=M2 seeds=1,2,3,4
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
tag = getarg(args, "tag", "")
case = anti_stokes_case(case_name)
Uˢ₀ = Float64(case.Uˢ₀)

dir(member; seed=0, extra=tag) = run_directory(root, case, level, member; seed, Δt, numerics, extra)
xmean(A) = dropdims(mean(A; dims=1); dims=1)
times_out = (2.0, 5.0, 10.0, 15.0, 20.0)

quiescent = dir("quiescent_control"; extra="")
isdir(quiescent) || (quiescent = nothing)
ΔU_waves, U_sh, U_sc, wrms, spectra = [], [], [], Dict{String, Vector{Any}}(), Dict{String, Vector{Any}}()
members_w = ("turbulence_control", "steady_waves_turbulence", "sheared_control", "steady_sheared_turbulence")
for m in members_w; wrms[m] = []; end
for m in ("steady_waves_turbulence", "sheared_control", "steady_sheared_turbulence"); spectra[m] = []; end
z = zf = t = nothing; k = 0.0
for seed in seeds
    sw, ct = dir("steady_waves_turbulence"; seed), dir("turbulence_control"; seed, extra="")
    ss, sc = dir("steady_sheared_turbulence"; seed), dir("sheared_control"; seed, extra="")
    all(isdir, (sw, ct, ss, sc)) || (@warn "seed $seed incomplete"; continue)
    r = load_run(sw; fields=("U", "W", "WW"))
    global t, z, zf, k = times(r), znodes_centers(r), znodes_faces(r), k₀(r)
    ΔU, _, _ = paired_residual(sw, ct, dir("steady_waves_null"), quiescent; fields=("U",))
    push!(ΔU_waves, xmean(ΔU))
    rs, rc = load_run(ss; fields=("U", "W", "WW")), load_run(sc; fields=("U", "W", "WW"))
    push!(U_sh, xmean(eulerian_U(rs))); push!(U_sc, xmean(xzt(rc, "U")))
    for (m, rr) in (("turbulence_control", load_run(ct; fields=("W", "WW"))), ("steady_waves_turbulence", r), ("sheared_control", rc), ("steady_sheared_turbulence", rs))
        W, WW = xzt(rr, "W"), xzt(rr, "WW")
        push!(wrms[m], sqrt.(max.(xmean(WW .- W .^ 2), 0)))
    end
    kδ = argmin(abs.(zf .+ 1 / (2k)))
    for (m, d) in (("steady_waves_turbulence", sw), ("sheared_control", sc), ("steady_sheared_turbulence", ss))
        wts = FieldTimeSeries(joinpath(d, "fov_plane.jld2"), "w")
        tt = collect(Float64, wts.times); Ny = size(interior(wts[1]), 2); Ly = r.meta["Ly"]
        S = Dict{Float64, Vector{Float64}}()
        for t₀ in times_out
            sel = findall(τt -> t₀ - 2.5 <= τt <= t₀ + 2.5, tt)
            acc = zeros(Ny ÷ 2 + 1)
            for n in sel
                wy = Float64.(vec(Array(interior(wts[n]))[1, :, kδ]))
                acc .+= abs2.(rfft(wy .- mean(wy))) ./ length(sel)
            end
            S[t₀] = acc
        end
        push!(spectra[m], (2π / Ly .* (0:Ny÷2), S))
    end
    @info @sprintf("seed %d: steady waves surface ΔU at t = 20 s: %.2f mm/s; sheared surface U at 20 s %.2f (control %.2f) mm/s",
                   seed, 1e3 * ΔU_waves[end][end, nearest_index(t, 20.0)], 1e3 * U_sh[end][end, nearest_index(t, 20.0)], 1e3 * U_sc[end][end, nearest_index(t, 20.0)])
end
n = length(ΔU_waves); n > 0 || error("no complete seeds")
kz = k .* z
avg(v) = mean(cat(v...; dims=3); dims=3)[:, :, 1]
ΔUw, Ush, Usc = avg(ΔU_waves), avg(U_sh), avg(U_sc)

# laminar seeded reference: growth of w_rms
lam = dir("steady_sheared_null")
lam_t = lam_w = nothing
if isdir(lam)
    f = jldopen(joinpath(lam, "statistics.jld2"))
    its = sort(parse.(Int, keys(f["timeseries/t"])))
    lam_t = [f["timeseries/t/$i"] for i in its]; lam_w = [f["timeseries/w_rms/$i"] for i in its]
    close(f)
    i1, i2 = nearest_index(lam_t, 4.0), nearest_index(lam_t, 12.0)
    @info @sprintf("Laminar seeded steady_sheared_null: w_rms growth rate between 4 and 12 s = %.3f s⁻¹, amplification to 20 s ×%.0f",
                   (log(lam_w[i2]) - log(lam_w[i1])) / (lam_t[i2] - lam_t[i1]), lam_w[nearest_index(lam_t, 20.0)] / lam_w[1])
end

set_theme!(Theme(fontsize=17))
fig = Figure(size=(2000, 1300))
Label(fig[0, 1:3], "Case $case_name at $level: steady wave train (Uˢ₀ = $(round(1e3Uˢ₀, digits=1)) mm/s, k = $(round(k, digits=1)) m⁻¹) switched on at t = 0 over the 1.D turbulence, without and with the initial current Uˢ₀e^{2kz}; $n seeds", fontsize=21)
colors = Makie.wong_colors()

ax1 = Axis(fig[1, 1]; xlabel="ΔU (mm/s)", ylabel="k₀ z", title="(1) steady waves, no shear: ΔU(z) since onset (dashed: −uˢ)")
for (j, t₀) in enumerate(times_out)
    lines!(ax1, 1e3 .* ΔUw[:, nearest_index(t, t₀)], kz; color=colors[j], linewidth=3, label="t = $(t₀) s")
end
lines!(ax1, -1e3 .* Uˢ₀ .* exp.(2kz), kz; color=(:gray, 0.6), linestyle=:dash)
vlines!(ax1, [0]; color=(:black, 0.3)); ylims!(ax1, -4, 0); axislegend(ax1; position=:lb, labelsize=13)

ax2 = Axis(fig[1, 2]; xlabel="t (s)", ylabel="mm/s", title="(2) surface Eulerian velocities against time")
lines!(ax2, t, 1e3 .* ΔUw[end, :]; color=colors[1], linewidth=3, label="ΔU, steady waves over turbulence")
lines!(ax2, t, 1e3 .* Ush[end, :]; color=colors[2], linewidth=3, label="⟨u⟩ᴱ, sheared + waves")
lines!(ax2, t, 1e3 .* Usc[end, :]; color=colors[2], linewidth=2, linestyle=:dash, label="⟨u⟩ᴱ, sheared control")
lines!(ax2, t, 1e3 .* (Ush[end, :] .- Usc[end, :]); color=colors[3], linewidth=3, label="difference (wave effect on the sheared current)")
hlines!(ax2, [0, -1e3Uˢ₀]; color=(:black, 0.3)); axislegend(ax2; position=:lb, labelsize=12)

ax3 = Axis(fig[1, 3]; xlabel="⟨u⟩ᴱ (mm/s)", ylabel="k₀ z", title="(3) sheared current: with waves (solid), control (dashed)")
for (j, t₀) in enumerate(times_out)
    ni = nearest_index(t, t₀)
    lines!(ax3, 1e3 .* Ush[:, ni], kz; color=colors[j], linewidth=2.5, label="t = $(t₀) s")
    lines!(ax3, 1e3 .* Usc[:, ni], kz; color=colors[j], linewidth=1.5, linestyle=:dash)
end
lines!(ax3, 1e3 .* Uˢ₀ .* exp.(2kz), kz; color=(:black, 0.4), linestyle=:dot, label="initial current")
vlines!(ax3, [0]; color=(:black, 0.3)); ylims!(ax3, -4, 0); axislegend(ax3; position=:rb, labelsize=12)

ax4 = Axis(fig[2, 1]; xlabel="w_rms (mm/s)", ylabel="k₀ z", title="(4) w_rms at t = 10 s (solid) and 20 s (dashed)")
for (j, m) in enumerate(members_w)
    Wm = avg(wrms[m]); kzw = k .* zf[1:size(Wm, 1)]
    lines!(ax4, 1e3 .* Wm[:, nearest_index(t, 10.0)], kzw; color=colors[j], linewidth=3, label=m)
    lines!(ax4, 1e3 .* Wm[:, nearest_index(t, 20.0)], kzw; color=colors[j], linewidth=2, linestyle=:dash)
end
ylims!(ax4, -4, 0); axislegend(ax4; position=:rb, labelsize=12)

ax5 = Axis(fig[2, 2]; xlabel="k_y (m⁻¹)", ylabel="spanwise spectrum of w at z = −δˢ", title="(5) spanwise w spectra at z = −δˢ (solid: sheared + waves; dotted: sheared control; dashed: waves only)", xscale=log10, yscale=log10)
for (j, t₀) in enumerate((5.0, 10.0, 20.0))
    for (m, ls) in (("steady_sheared_turbulence", :solid), ("sheared_control", :dot), ("steady_waves_turbulence", :dash))
        Sm = mean(hcat([s[2][t₀] for s in spectra[m]]...); dims=2) |> vec
        ky = spectra[m][1][1]
        lines!(ax5, ky[2:end], Sm[2:end] .+ 1e-30; color=colors[j], linewidth=ls == :solid ? 3 : 1.5, linestyle=ls, label=ls == :solid ? "t = $(t₀) s" : nothing)
    end
end
vlines!(ax5, [2k]; color=(:black, 0.4), linestyle=:dot, label="2k₀")
axislegend(ax5; position=:lb, labelsize=12)

ax6 = Axis(fig[2, 3]; xlabel="t (s)", ylabel="w_rms (mm/s)", title="(6) laminar seeded reference: CL2 growth of w_rms", yscale=log10)
if !isnothing(lam_t)
    lines!(ax6, lam_t, 1e3 .* lam_w; color=:black, linewidth=3)
    i1, i2 = nearest_index(lam_t, 4.0), nearest_index(lam_t, 12.0)
    σ = (log(lam_w[i2]) - log(lam_w[i1])) / (lam_t[i2] - lam_t[i1])
    lines!(ax6, lam_t, 1e3 .* lam_w[i1] .* exp.(σ .* (lam_t .- lam_t[i1])); color=:red, linestyle=:dash, label=@sprintf("e-folding %.2f s⁻¹", σ))
    axislegend(ax6; position=:lt)
end

output = get(args, "output", joinpath(figure_directory(), "steady_waves_channel_" * replace(case_name, "." => "") * "_$(level)_$(numerics)$(isempty(tag) ? "" : "_" * tag).png"))
save(output, fig)
@info "Saved $output"
