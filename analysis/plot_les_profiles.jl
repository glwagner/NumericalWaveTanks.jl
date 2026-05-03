using CairoMakie
using Oceananigans
using JLD2
using Printf

# Hovmöller plots of LES horizontally-averaged profiles —
# the primary diagnostics from research_objectives.md.
#
# Usage:
#   julia --project analysis/plot_les_profiles.jl <run_dir>

run_dir = ARGS[1]
prefix  = basename(rstrip(run_dir, '/'))
profiles_file = joinpath(run_dir, prefix * "_profiles.jld2")
isfile(profiles_file) || error("missing $profiles_file")

@info "Loading $profiles_file"
U  = FieldTimeSeries(profiles_file, "U")
ww = FieldTimeSeries(profiles_file, "ww")
uu = FieldTimeSeries(profiles_file, "uu")
uw = FieldTimeSeries(profiles_file, "uw")

times = U.times
zs   = znodes(U)        # Center
zs_w = znodes(ww)       # Face — one extra cell
@info "Loaded $(length(times)) snapshots; t = $(first(times)) → $(last(times)) s"
@info "Nz_center = $(length(zs)), Nz_face = $(length(zs_w))"

Nt = length(times)
Nz = length(zs)
Nzw = length(zs_w)
U_arr  = zeros(Float32, Nz,  Nt)
uu_arr = zeros(Float32, Nz,  Nt)
uw_arr = zeros(Float32, Nz,  Nt)
ww_arr = zeros(Float32, Nzw, Nt)   # at Face
for n in 1:Nt
    U_arr[:, n]  = vec(interior(U[n]))
    uu_arr[:, n] = vec(interior(uu[n]))
    uw_arr[:, n] = vec(interior(uw[n]))
    ww_arr[:, n] = vec(interior(ww[n]))
end

# Reynolds-stress recovery: <u'²> = <u²> - U², similar for u'w'
up2 = uu_arr .- U_arr.^2
uw_prime = uw_arr .- U_arr .* 0  # <w>=0, so <u'w'> = <u w>

@info @sprintf("U range:    min=%.3e, max=%.3e", minimum(U_arr), maximum(U_arr))
@info @sprintf("<w²>:       min=%.3e, max=%.3e", minimum(ww_arr), maximum(ww_arr))
@info @sprintf("<u'²>:      min=%.3e, max=%.3e", minimum(up2), maximum(up2))
@info @sprintf("<u'w'>:     min=%.3e, max=%.3e", minimum(uw_prime), maximum(uw_prime))

fig = Figure(size=(1500, 950), fontsize=15)

# Panel a: U(z, t)
ax_U = Axis(fig[1, 1]; xlabel="t (s)", ylabel="z (m)", title="U(z, t)  [m/s]")
hm_U = heatmap!(ax_U, times, zs, U_arr'; colormap=:thermal)
Colorbar(fig[1, 2], hm_U; label="U (m/s)", width=15)

# Panel b: <w²>(z, t) — w lives at Face in z
ax_ww = Axis(fig[1, 3]; xlabel="t (s)", ylabel="", title="⟨w²⟩(z, t)  [m²/s²]")
hm_ww = heatmap!(ax_ww, times, zs_w, ww_arr'; colormap=:viridis)
Colorbar(fig[1, 4], hm_ww; label="⟨w²⟩", width=15)

# Panel c: <u'²>(z, t)
ax_up = Axis(fig[2, 1]; xlabel="t (s)", ylabel="z (m)", title="⟨u'²⟩(z, t)  [m²/s²]")
hm_up = heatmap!(ax_up, times, zs, up2'; colormap=:viridis)
Colorbar(fig[2, 2], hm_up; label="⟨u'²⟩", width=15)

# Panel d: <u'w'>(z, t)
ax_uw = Axis(fig[2, 3]; xlabel="t (s)", ylabel="", title="⟨u'w'⟩(z, t)  [m²/s²]")
uw_max = maximum(abs, uw_prime)
hm_uw = heatmap!(ax_uw, times, zs, uw_prime'; colormap=:balance, colorrange=(-uw_max, uw_max))
Colorbar(fig[2, 4], hm_uw; label="⟨u'w'⟩", width=15)

Label(fig[0, 1:4], "LES horizontally-averaged profiles — $prefix"; fontsize=14)

out = joinpath(run_dir, "profiles_hovmoller.png")
save(out, fig; px_per_unit=2)
@info "Saved $out"

# Also: end-state profiles vs depth
fig2 = Figure(size=(1100, 800), fontsize=15)
ax1 = Axis(fig2[1, 1]; xlabel="U (m/s)", ylabel="z (m)", title="Mean streamwise velocity")
ax2 = Axis(fig2[1, 2]; xlabel="⟨w²⟩ (m²/s²)", ylabel="", title="Vertical-velocity variance")
ax3 = Axis(fig2[2, 1]; xlabel="⟨u'²⟩ (m²/s²)", ylabel="z (m)", title="Streamwise-velocity variance")
ax4 = Axis(fig2[2, 2]; xlabel="⟨u'w'⟩ (m²/s²)", ylabel="", title="Reynolds momentum flux")

# Plot at four times
nts = round.(Int, range(1, Nt, length=4))
for n in nts
    lines!(ax1, U_arr[:, n], zs; label=@sprintf("t = %.1f s", times[n]))
    lines!(ax2, ww_arr[:, n], zs_w)
    lines!(ax3, up2[:, n], zs)
    lines!(ax4, uw_prime[:, n], zs)
end
axislegend(ax1; position=:rb, fontsize=10)

Label(fig2[0, 1:2], "LES mean profiles — $prefix"; fontsize=14)

out2 = joinpath(run_dir, "profiles_vs_depth.png")
save(out2, fig2; px_per_unit=2)
@info "Saved $out2"
