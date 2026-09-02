#####
##### The decisive paired figure (campaign document, sections 9.6, 11.3, 11.5, 11.6, 19):
#####
#####   ΔU_turb = (U_packet+turb − U_turb) − (U_packet+null − U_null)
#####
#####   1. Hovmöller of surface ΔU_turb showing the packet crossing the FOV and its wake
#####   2. the packet-only Eulerian return flow
#####   3. the post-packet ΔU(z) profile at the FOV against k₀z (with a wake-age estimate)
#####   4. the associated change in ⟨u'w'⟩
#####   5. FOV time series: surface ΔU and depth-integrated transport
#####   6. slope ratio R = −∂zΔU / ∂zuˢ_peak against the anisotropy A = u'²/w'²
#####
##### Usage: julia --project=. analysis/anti_stokes/compare_packet_control.jl packet=<dir> control=<dir>
#####            [null=<dir> quiescent=<dir> output=<png>]
#####

using CairoMakie
include("quick_checks.jl")

args = parse_key_value_args(ARGS)
packet_dir, control_dir = args["packet"], args["control"]
null_dir = get(args, "null", nothing)
quiescent_dir = get(args, "quiescent", nothing)

report = pair_report(packet_dir, control_dir, null_dir, quiescent_dir)
ΔU, pk, ct = paired_residual(packet_dir, control_dir, null_dir, quiescent_dir)

p, c = run_packet(pk), run_case(pk)
t, x, z, zf, Δz = times(pk), xnodes_faces(pk), znodes_centers(pk), znodes_faces(pk), Δz_centers(pk)
τ, tp, i, k = τ₀(pk), t_peak(pk), fov_index(pk), k₀(pk)
Lx, Nz = pk.meta["Lx"], length(z)

# Pairing check: both members must come from the same checkpoint
pk.meta["initial_condition_sha256"] == ct.meta["initial_condition_sha256"] ||
    @warn "Packet and control members do not share an initial-condition checksum"

w = windows(pk)
before = window_mean(ΔU, t, w.before...)[i, :]
after  = window_mean(ΔU, t, w.after...)[i, :]
profile = after .- before

# Wake-age estimate at the final time: columns passed 3–4 widths ago
xc_end = packet_center(t[end], p)   # evaluated outside @. so the NamedTuple p is not broadcast
age = @. mod(xc_end - x, Lx)
wake = findall(a -> 3p.σ₀ <= a <= 4p.σ₀, age)
wake_profile = isempty(wake) ? fill(NaN, Nz) : dropdims(mean(ΔU[wake, :, end]; dims=1); dims=1) .- before

m_pk, m_ct = central_moments(pk), central_moments(ct)
Δuw = m_pk.uw .- m_ct.uw
Δuw_before = window_mean(Δuw, t, w.before...)[i, :]
Δuw_after  = window_mean(Δuw, t, w.after...)[i, :]

transport_t = [sum(ΔU[i, :, n] .* Δz) for n in eachindex(t)]

# Packet-only Eulerian response (return flow) at the surface
U_null_surface = nothing
if !isnothing(null_dir)
    nl = load_run(null_dir; fields=("U",))
    U_null_surface = eulerian_U(nl)[:, end, :]
    isnothing(quiescent_dir) || (U_null_surface .-= xzt(load_run(quiescent_dir; fields=("U",)), "U")[:, end, :])
end

# Wake-age composite (low-noise estimate using the whole wake)
ages, Cw, Nw = wake_age_composite(ΔU, x, t, p; age_edges=default_age_edges(τ))
composite_after = composite_profile(ages, Cw, τ, 2.125, 3.875)

# Quasi-equilibrium ratio R = −∂zΔU / ∂zuˢ_peak from the composite profile (age 2–4 τ₀) against
# the anisotropy A = u'²/w'² of the control, x-averaged over the after window
∂zΔU = diff(composite_after) ./ diff(z)
∂zuˢ = [2p.k * p.Uˢ₀ * exp(2p.k * zk) for zk in zf[2:end-1]]
R = -∂zΔU ./ ∂zuˢ
uu_ct = vec(mean(window_mean(m_ct.uu, t, w.after...); dims=1))
ww_ct = vec(mean(window_mean(m_ct.ww, t, w.after...); dims=1))
A = uu_ct[2:end] ./ max.(ww_ct[2:end-1], 1e-12)

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1800, 2000))
Label(fig[0, 1:4], @sprintf("Case %s, %s, seed %d: %s − %s%s", c.name, pk.meta["level"], pk.meta["seed"],
                            pk.meta["member"], ct.meta["member"], isnothing(null_dir) ? "" : ", null-corrected"), fontsize=22)

xc = [mod(packet_center(tn, p), Lx) for tn in t]

ax1 = Axis(fig[1, 1]; xlabel="x (m)", ylabel="t (s)", title="(1) surface ΔU_turb (mm/s)")
hm1 = heatmap!(ax1, x, t, 1e3 .* ΔU[:, end, :]; colormap=:balance, colorrange=1e3 .* symmetric_range(ΔU[:, end, :]))
lines!(ax1, xc, t; color=:black, linestyle=:dash)
vlines!(ax1, [p.x_FOV]; color=:black)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(fig[1, 3]; xlabel="x (m)", ylabel="t (s)", title="(2) packet-only Eulerian response at the surface (mm/s)")
if isnothing(U_null_surface)
    text!(ax2, 0.5, 0.5; text="no packet-null run supplied", space=:relative, align=(:center, :center))
else
    hm2 = heatmap!(ax2, x, t, 1e3 .* U_null_surface; colormap=:balance, colorrange=1e3 .* symmetric_range(U_null_surface))
    lines!(ax2, xc, t; color=:black, linestyle=:dash)
    vlines!(ax2, [p.x_FOV]; color=:black)
    Colorbar(fig[1, 4], hm2)
end

ax3 = Axis(fig[2, 1:2]; xlabel="ΔU (mm/s)", ylabel="k₀ z", title="(3) residual at the observation plane")
lines!(ax3, 1e3 .* before, k .* z; label="before (t ≤ t_peak − 3τ₀)", color=:gray)
lines!(ax3, 1e3 .* after, k .* z; label="after (t_peak + 3τ₀ ≤ t ≤ t_peak + 4τ₀)", color=:orange)
lines!(ax3, 1e3 .* profile, k .* z; label="after − before", color=:black, linewidth=3)
lines!(ax3, 1e3 .* wake_profile, k .* z; label="wake age 3–4σ₀ at t_stop", color=:purple, linestyle=:dash)
vlines!(ax3, [0]; color=(:black, 0.3))
ylims!(ax3, -4, 0)
axislegend(ax3; position=:rb)

ax4 = Axis(fig[2, 3:4]; xlabel="Δ⟨u'w'⟩ (10⁻⁶ m²/s²)", ylabel="k₀ z", title="(4) Reynolds-stress change at the FOV")
lines!(ax4, 1e6 .* Δuw_before, k .* z; label="before", color=:gray)
lines!(ax4, 1e6 .* Δuw_after, k .* z; label="after", color=:orange)
lines!(ax4, 1e6 .* (Δuw_after .- Δuw_before), k .* z; label="after − before", color=:black, linewidth=3)
lines!(ax4, 1e6 .* window_mean(Δuw, t, w.passage...)[i, :], k .* z; label="passage (t_peak ± τ₀)", color=:firebrick, linestyle=:dot)
vlines!(ax4, [0]; color=(:black, 0.3))
ylims!(ax4, -4, 0)
axislegend(ax4; position=:rb)

ax5 = Axis(fig[3, 1:2]; xlabel="t (s)", ylabel="mm/s   |   10⁻³ m²/s", title="(5) observation-plane time series")
lines!(ax5, t, 1e3 .* ΔU[i, end, :]; label="surface ΔU_turb (mm/s)", linewidth=2)
lines!(ax5, t, 1e3 .* transport_t; label="∫ΔU dz (10⁻³ m²/s)", linewidth=2)
lines!(ax5, t, 1e3 .* [uˢ(x[i], 0, 0, tn, p) for tn in t]; color=(:gray, 0.6), linestyle=:dash, label="uˢ envelope (mm/s)")
vspan!(ax5, [w.before[1], w.after[1]], [w.before[2], w.after[2]]; color=(:green, 0.1))
axislegend(ax5; position=:lt)

ax6 = Axis(fig[3, 3:4]; xlabel="ratio", ylabel="k₀ z", title="(6) R = −∂zΔU/∂zuˢ_peak (composite, age 2–4 τ₀) vs A = u'²/w'² (control, x-averaged)")
lines!(ax6, R, k .* zf[2:end-1]; label="R (composite)", color=:black, linewidth=3)
lines!(ax6, A, k .* zf[2:end-1]; label="A (control)", color=:red)
xlims!(ax6, -1, 5)
ylims!(ax6, -4, 0)
axislegend(ax6; position=:rb)

ax7 = Axis(fig[4, 1]; xlabel="age (x_c(t) − x) / cᵍ  (units of τ₀)", ylabel="k₀ z",
           title="(7) wake-age composite of ΔU_turb (mm/s), all x and t")
hm7 = heatmap!(ax7, ages ./ τ, k .* z, 1e3 .* permutedims(Cw); colormap=:balance, colorrange=1e3 .* symmetric_range(Cw))
vlines!(ax7, [0]; color=:black, linestyle=:dash)
ylims!(ax7, -4, 0)
Colorbar(fig[4, 2], hm7)

ax8 = Axis(fig[4, 3:4]; xlabel="ΔU (mm/s)", ylabel="k₀ z", title="(8) composite profiles by age")
for (a₀, a₁, color) in ((0, 1, :gray), (1, 2, :orange), (2, 3, :firebrick), (3, 4, :purple), (4, 6, :navy))
    lines!(ax8, 1e3 .* composite_profile(ages, Cw, τ, a₀ + 0.125, a₁ - 0.125), k .* z; label="age $(a₀)–$(a₁) τ₀", color, linewidth=2)
end
lines!(ax8, 1e3 .* profile, k .* z; label="FOV after − before", color=:black, linestyle=:dash)
vlines!(ax8, [0]; color=(:black, 0.3))
ylims!(ax8, -4, 0)
axislegend(ax8; position=:rb)

output = get(args, "output", joinpath(figure_directory(),
             "paired_residual_$(case_dirname(c))_$(pk.meta["level"])$(is_bounded_x(pk) ? "_boundedx" : "")_seed$(pk.meta["seed"])$(isnothing(null_dir) ? "" : "_nullcorrected").png"))
save(output, fig)
@info "Saved $output"
