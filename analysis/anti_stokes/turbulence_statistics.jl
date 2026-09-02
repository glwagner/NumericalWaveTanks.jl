#####
##### Turbulence statistics of a no-wave control (campaign document, 8.3 and 11.7):
#####
#####   (a) rms profiles at t = 0, t_peak, t_stop against the case targets
#####   (b) anisotropy u'²/w'² profile at t_peak
#####   (c) volume rms time series against the targets
#####   (d) streamwise integral scale profile from the 3D snapshot at 4τ₀
#####   (e) streamwise spectrum of u at mid-depth and near the surface from the same snapshot
#####   (f) ⟨u'w'⟩ profiles
#####
##### Optionally overlays a packet member (packet=<dir>) on (a), (b), and (f).
##### Usage: julia --project=. analysis/anti_stokes/turbulence_statistics.jl control=<dir> [packet=<dir>] [output=<png>]
#####

using CairoMakie
using FFTW
include("quick_checks.jl")

args = parse_key_value_args(ARGS)
control_dir = args["control"]
packet_dir = get(args, "packet", nothing)

report = turbulence_report(control_dir)

ct = load_run(control_dir)
c = run_case(ct)
t, z, zf = times(ct), znodes_centers(ct), znodes_faces(ct)
tp, τ, k = t_peak(ct), τ₀(ct), k₀(ct)
stats = load_statistics(control_dir)
m = central_moments(ct)

x_average(A) = dropdims(mean(A; dims=1); dims=1)  # (Nz, Nt)
uu, vv, ww, uw = x_average(m.uu), x_average(m.vv), x_average(m.ww), x_average(m.uw)
wwc = 0.5 .* (ww[1:end-1, :] .+ ww[2:end, :])
rms(a) = sqrt.(max.(a, 0))

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1500, 1500))
Label(fig[0, 1:3], "Turbulence control, case $(c.name), $(ct.meta["level"]), seed $(ct.meta["seed"])", fontsize=22)

ax_a = Axis(fig[1, 1]; xlabel="rms velocity (mm/s)", ylabel="k₀ z", title="(a) rms profiles (x-averaged)")
for (label, n, style) in (("t = 0", 1, :dot), ("t_peak", nearest_index(t, tp), :solid), ("t_stop", length(t), :dash))
    lines!(ax_a, 1e3 .* rms(uu[:, n]), k .* z; color=:royalblue, linestyle=style, label="u ($label)")
    lines!(ax_a, 1e3 .* rms(vv[:, n]), k .* z; color=:seagreen, linestyle=style, label="v ($label)")
    lines!(ax_a, 1e3 .* rms(wwc[:, n]), k .* z; color=:firebrick, linestyle=style, label="w ($label)")
end
vlines!(ax_a, 1e3 .* [Float64(c.u_rms)]; color=:royalblue, linestyle=:dashdot)
vlines!(ax_a, 1e3 .* [Float64(c.w_rms)]; color=:firebrick, linestyle=:dashdot)
axislegend(ax_a; position=:rb, nbanks=2, labelsize=11)

ax_b = Axis(fig[1, 2]; xlabel="u'² / w'²", ylabel="k₀ z", title="(b) anisotropy at t_peak", xscale=log10)
n_peak = nearest_index(t, tp)
lines!(ax_b, max.(uu[:, n_peak] ./ max.(wwc[:, n_peak], 1e-12), 1e-3), k .* z; label="control", linewidth=3)
lines!(ax_b, max.(vv[:, n_peak] ./ max.(wwc[:, n_peak], 1e-12), 1e-3), k .* z; label="v'²/w'²", linewidth=1)
vlines!(ax_b, [(Float64(c.u_rms) / Float64(c.w_rms))^2]; color=:black, linestyle=:dash, label="measured bulk")
axislegend(ax_b; position=:rb)

ax_c = Axis(fig[1, 3]; xlabel="t (s)", ylabel="volume rms (mm/s)", title="(c) rms decay")
lines!(ax_c, stats["t"], 1e3 .* stats["u_rms"]; label="u", color=:royalblue, linewidth=2)
lines!(ax_c, stats["t"], 1e3 .* stats["v_rms"]; label="v", color=:seagreen, linewidth=2)
lines!(ax_c, stats["t"], 1e3 .* stats["w_rms"]; label="w", color=:firebrick, linewidth=2)
hlines!(ax_c, 1e3 .* [Float64(c.u_rms)]; color=:royalblue, linestyle=:dash)
hlines!(ax_c, 1e3 .* [Float64(c.w_rms)]; color=:firebrick, linestyle=:dash)
vlines!(ax_c, [tp - 3τ, tp, tp + 3τ]; color=(:gray, 0.6))
axislegend(ax_c; position=:rt)

if isfile(joinpath(control_dir, "snapshots.jld2"))
    u3, ts = load_snapshot(control_dir, "u", tp)
    Nx = size(u3, 1)
    Δx = ct.meta["Lx"] / Nx
    L = integral_scale_profile(u3, Δx)

    ax_d = Axis(fig[2, 1]; xlabel="L₁₁ (m)", ylabel="k₀ z", title="(d) streamwise integral scale, t = $(round(ts, digits=2)) s")
    lines!(ax_d, L, k .* z; linewidth=3)
    vlines!(ax_d, [Float64(c.L)]; color=:black, linestyle=:dash, label="target")
    axislegend(ax_d; position=:rb)

    kx = 2π / ct.meta["Lx"] .* (1:Nx÷2)
    function spectrum(k_index)
        û = rfft(u3[:, :, k_index], 1)
        E = dropdims(mean(abs2, û; dims=2); dims=2)[2:Nx÷2+1] .* (2Δx / Nx)
        return E
    end
    ax_e = Axis(fig[2, 2]; xlabel="kₓ (m⁻¹)", ylabel="E(kₓ)", xscale=log10, yscale=log10, title="(e) streamwise u spectrum")
    for (label, kk) in (("mid-depth", nearest_index(z, -0.2)), ("top cell", length(z)), ("δˢ", nearest_index(z, -Float64(c.δˢ))))
        E = spectrum(kk)
        lines!(ax_e, kx, max.(E, 1e-30); label, linewidth=2)
    end
    ref = 1e-4 .* (kx ./ kx[10]) .^ (-5/3)
    lines!(ax_e, kx, ref; color=:black, linestyle=:dash, label="k⁻⁵ᐟ³")
    vlines!(ax_e, [2π / Float64(c.L), 2 * Float64(c.k), π / Δx]; color=(:gray, 0.6))
    axislegend(ax_e; position=:rt)
end

ax_f = Axis(fig[2, 3]; xlabel="⟨u'w'⟩ (mm²/s²)", ylabel="k₀ z", title="(f) Reynolds stress (x-averaged)")
for (label, n, style) in (("t = 0", 1, :dot), ("t_peak", n_peak, :solid), ("t_stop", length(t), :dash))
    lines!(ax_f, 1e6 .* uw[:, n], k .* z; color=:royalblue, linestyle=style, label="control, $label")
end
if !isnothing(packet_dir)
    pk = load_run(packet_dir)
    mp = central_moments(pk)
    uwp = x_average(mp.uw)
    for (label, n, style) in (("t_peak", n_peak, :solid), ("t_stop", length(t), :dash))
        lines!(ax_f, 1e6 .* uwp[:, n], k .* z; color=:firebrick, linestyle=style, label="packet, $label")
    end
    uup, wwp = x_average(mp.uu), x_average(mp.ww)
    wwpc = 0.5 .* (wwp[1:end-1, :] .+ wwp[2:end, :])
    lines!(ax_a, 1e3 .* rms(uup[:, n_peak]), k .* z; color=:black, linestyle=:solid, label="u (packet, t_peak)")
    lines!(ax_b, max.(uup[:, n_peak] ./ max.(wwpc[:, n_peak], 1e-12), 1e-3), k .* z; color=:firebrick, label="packet")
end
vlines!(ax_f, [0]; color=:black)
axislegend(ax_f; position=:rb, labelsize=11)

output = get(args, "output", joinpath(figure_directory(), "turbulence_statistics_$(case_dirname(c))_$(ct.meta["level"])_seed$(ct.meta["seed"]).png"))
save(output, fig)
@info "Saved $output"
