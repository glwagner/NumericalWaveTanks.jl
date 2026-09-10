#####
##### Fourth case: steady wave train + 1.D turbulence + initial current α Uˢ₀ e^{2kz} + constant surface
##### momentum flux u*² (wind_sheared_turbulence) against the same without waves (wind_sheared_control),
##### with the laminar seeded reference wind_sheared_null. Profiles are x- and y-averaged; the surface
##### anisotropy of w (streamwise-elongated fraction) is computed from the seed-1 xy slices.
#####
##### Usage: sbatch batch/anti_stokes_analysis.batch script=wind_channel.jl case=1.D level=M2 seeds=1,2,3,4 tag=ustar4.5
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
tag = getarg(args, "tag", "ustar4.5")
case = anti_stokes_case(case_name)
Uˢ₀ = Float64(case.Uˢ₀)
dir(member; seed=0, extra=tag) = run_directory(root, case, level, member; seed, Δt, numerics, extra)
xmean(A) = dropdims(mean(A; dims=1); dims=1)

U_w, U_c, ΔU, wrms_w, wrms_c, uw_w, uw_c = [], [], [], [], [], [], []
z = zf = t = nothing; k = 0.0; τ = 0.0
for seed in seeds
    dw, dc = dir("wind_sheared_turbulence"; seed), dir("wind_sheared_control"; seed)
    all(isdir, (dw, dc)) || (@warn "seed $seed incomplete"; continue)
    rw, rc = load_run(dw; fields=("U", "W", "WW", "UW")), load_run(dc; fields=("U", "W", "WW", "UW"))
    global t, z, zf, k = times(rw), znodes_centers(rw), znodes_faces(rw), k₀(rw)
    global τ = rw.meta["wind_stress"]
    push!(U_w, xmean(eulerian_U(rw))); push!(U_c, xmean(xzt(rc, "U")))
    push!(ΔU, U_w[end] .- U_c[end])
    for (r, acc_w, acc_uw) in ((rw, wrms_w, uw_w), (rc, wrms_c, uw_c))
        W, WW = xzt(r, "W"), xzt(r, "WW")
        push!(acc_w, sqrt.(max.(xmean(WW .- W .^ 2), 0)))
        m = central_moments(r)
        push!(acc_uw, xmean(m.uw))
    end
    @info @sprintf("seed %d: surface ⟨u⟩ᴱ at 20/40/60 s with waves %.1f/%.1f/%.1f mm/s, control %.1f/%.1f/%.1f mm/s", seed,
                   (1e3 .* U_w[end][end, nearest_index.(Ref(t), (20.0, 40.0, min(60.0, t[end])))])..., (1e3 .* U_c[end][end, nearest_index.(Ref(t), (20.0, 40.0, min(60.0, t[end])))])...)
end
n = length(U_w); n > 0 || error("no complete seeds")
avg(v) = mean(cat(v...; dims=3); dims=3)[:, :, 1]
Uw, Uc, ΔUm, Ww, Wc = avg(U_w), avg(U_c), avg(ΔU), avg(wrms_w), avg(wrms_c)
kz = k .* z
ustar = sqrt(τ)

# surface anisotropy from seed-1 xy slices (with waves, control, laminar null)
function elongation(d)
    W = FieldTimeSeries(joinpath(d, "xy_surface.jld2"), "w")
    tt = collect(Float64, W.times); grid = W.grid
    Nx, Ny = size(interior(W[1]))[1:2]
    kx = 2π / grid.Lx .* fftfreq(Nx, Nx); ky = 2π / grid.Ly .* fftfreq(Ny, Ny)
    elong = [abs(kx[i]) < abs(ky[j]) / 3 && ky[j] != 0 for i in 1:Nx, j in 1:Ny]
    frac = Float64[]; wr = Float64[]
    for nn in 1:2:length(tt)
        w = Float64.(Array(interior(W[nn]))[:, :, 1]); w .-= mean(w)
        F = abs2.(fft(w)); F[1, 1] = 0
        push!(frac, sum(F[elong]) / sum(F)); push!(wr, sqrt(mean(abs2, w)))
    end
    return tt[1:2:end], frac, wr
end
el = Dict()
for (m, d) in (("with waves", dir("wind_sheared_turbulence"; seed=first(seeds))), ("control", dir("wind_sheared_control"; seed=first(seeds))), ("laminar null", dir("wind_sheared_null")))
    isdir(d) && isfile(joinpath(d, "xy_surface.jld2")) && (el[m] = elongation(d))
end

set_theme!(Theme(fontsize=17))
fig = Figure(size=(2000, 1300))
Label(fig[0, 1:3], @sprintf("Case %s at %s: steady waves (Uˢ₀ = %.1f mm/s) + turbulence + initial current Uˢ₀e^{2kz} + wind stress u*² = %.1e m²/s² (u* = %.1f mm/s, La_t = %.2f); %d seeds",
                            case_name, level, 1e3Uˢ₀, τ, 1e3ustar, sqrt(ustar / Uˢ₀), n), fontsize=20)
colors = Makie.wong_colors()
tsel = filter(<=(t[end] + 1e-6), (5.0, 10.0, 20.0, 40.0, 60.0))

ax1 = Axis(fig[1, 1]; xlabel="⟨u⟩ᴱ (mm/s)", ylabel="k₀ z", title="(1) Eulerian current: with waves (solid), wind-only control (dashed)")
for (j, t₀) in enumerate(tsel)
    ni = nearest_index(t, t₀)
    lines!(ax1, 1e3 .* Uw[:, ni], kz; color=colors[j], linewidth=2.5, label="t = $(t₀) s")
    lines!(ax1, 1e3 .* Uc[:, ni], kz; color=colors[j], linewidth=1.5, linestyle=:dash)
end
lines!(ax1, 1e3 .* Uˢ₀ .* exp.(2kz), kz; color=(:black, 0.4), linestyle=:dot, label="initial current")
vlines!(ax1, [0]; color=(:black, 0.3)); ylims!(ax1, -4, 0); axislegend(ax1; position=:rb, labelsize=12)

ax2 = Axis(fig[1, 2]; xlabel="t (s)", ylabel="mm/s", title="(2) surface Eulerian velocity and the wave effect")
lines!(ax2, t, 1e3 .* Uw[end, :]; color=colors[2], linewidth=3, label="with waves")
lines!(ax2, t, 1e3 .* Uc[end, :]; color=colors[2], linewidth=2, linestyle=:dash, label="control (shear + wind)")
lines!(ax2, t, 1e3 .* ΔUm[end, :]; color=colors[3], linewidth=3, label="difference")
lines!(ax2, t, 1e3 .* (Uˢ₀ .+ τ .* t ./ (1 / (2k))); color=(:gray, 0.6), linestyle=:dot, label="initial + u*² t / δˢ (no mixing)")
hlines!(ax2, [0]; color=(:black, 0.3)); axislegend(ax2; position=:lb, labelsize=12)

ax3 = Axis(fig[1, 3]; xlabel="ΔU = ⟨u⟩ᴱ_waves − ⟨u⟩ᴱ_control (mm/s)", ylabel="k₀ z", title="(3) wave-induced change of the wind-driven current")
for (j, t₀) in enumerate(tsel)
    lines!(ax3, 1e3 .* ΔUm[:, nearest_index(t, t₀)], kz; color=colors[j], linewidth=2.5, label="t = $(t₀) s")
end
lines!(ax3, -1e3 .* Uˢ₀ .* exp.(2kz), kz; color=(:gray, 0.6), linestyle=:dash, label="−uˢ")
vlines!(ax3, [0]; color=(:black, 0.3)); ylims!(ax3, -4, 0); axislegend(ax3; position=:lb, labelsize=12)

ax4 = Axis(fig[2, 1]; xlabel="w_rms (mm/s)", ylabel="k₀ z", title="(4) w_rms: with waves (solid), control (dashed)")
for (j, t₀) in enumerate(tsel)
    ni = nearest_index(t, t₀); kzw = k .* zf[1:size(Ww, 1)]
    lines!(ax4, 1e3 .* Ww[:, ni], kzw; color=colors[j], linewidth=2.5, label="t = $(t₀) s")
    lines!(ax4, 1e3 .* Wc[:, ni], kzw; color=colors[j], linewidth=1.5, linestyle=:dash)
end
ylims!(ax4, -4, 0); axislegend(ax4; position=:rb, labelsize=12)

ax5 = Axis(fig[2, 2]; xlabel="t (s)", ylabel="fraction of surface w energy in streamwise-elongated modes (|k_x| < |k_y|/3)", title="(5) surface anisotropy of w (seed 1); Langmuir cells → high fraction")
for (j, (m, v)) in enumerate(el)
    lines!(ax5, v[1], v[2]; color=colors[j], linewidth=2.5, label=m)
end
hlines!(ax5, [0.27]; color=(:black, 0.3), linestyle=:dot, label="plain turbulence (0.27)")
axislegend(ax5; position=:rt, labelsize=12)

ax6 = Axis(fig[2, 3]; xlabel="t (s)", ylabel="surface w_rms (mm/s, top cell)", title="(6) surface w_rms (seed 1) and the laminar seeded reference", yscale=log10)
for (j, (m, v)) in enumerate(el)
    lines!(ax6, v[1], 1e3 .* max.(v[3], 1e-9); color=colors[j], linewidth=2.5, label=m)
end
axislegend(ax6; position=:rb, labelsize=12)

output = get(args, "output", joinpath(figure_directory(), "wind_channel_" * replace(case_name, "." => "") * "_$(level)_$(numerics)_$(tag).png"))
save(output, fig)
@info "Saved $output"
