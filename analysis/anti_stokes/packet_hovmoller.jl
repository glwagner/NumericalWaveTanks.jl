#####
##### Packet trajectory and Hovmöller diagnostics for one run (campaign document, 11.1–11.2):
#####
#####   (a) prescribed surface Stokes drift at the FOV: sampled vs analytic envelope
#####   (b) ⟨u^L⟩_y at the surface, x–t
#####   (c) ⟨u^E⟩_y = ⟨u^L⟩_y − uˢ at the surface, x–t (minus a control run if given)
#####   (d) the same in packet coordinates ξ = x − x_c(t)
#####   (e) ⟨u^E⟩_y(z) at the FOV at the analysis times
#####   (f) ⟨u^E⟩_y at the FOV, z–t
#####
##### Usage: julia --project=. analysis/anti_stokes/packet_hovmoller.jl run=<dir> [control=<dir>] [output=<png>]
#####

using CairoMakie
include("common.jl")

args = parse_key_value_args(ARGS)
run_dir = args["run"]
control_dir = get(args, "control", nothing)

run = load_run(run_dir; fields=("U", "W"))
p = run_packet(run)
t, x, z = times(run), xnodes_faces(run), znodes_centers(run)
τ, tp, i_fov, k = τ₀(run), t_peak(run), fov_index(run), k₀(run)
Lx, Nx = run.meta["Lx"], run.meta["Nx"]
Δx = Lx / Nx

UL = xzt(run, "U")
UE = eulerian_U(run)
label = "⟨u^E⟩_y = ⟨u^L⟩_y − uˢ"
if !isnothing(control_dir)
    control = load_run(control_dir; fields=("U",))
    UE .-= xzt(control, "U")
    label = "⟨u^E⟩_y − control"
end

stats = load_statistics(run_dir)
analytic = @. p.Uˢ₀ * exp(-((stats["t"] - tp) / τ)^2)

# Packet coordinates: Uξ[j, n] = UE at x = ξ_j + x_c(t_n), with ξ_j = x_j − Lx/2
ξ = x .- Lx / 2
Uξ = similar(UE[:, end, :])
for n in eachindex(t)
    shift = round(Int, (packet_center(t[n], p) - Lx / 2) / Δx)
    for j in eachindex(x)
        Uξ[j, n] = UE[mod1(j + shift, Nx), end, n]
    end
end

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1500, 1700))
member = run.meta["member"]
Label(fig[0, 1:2], "Case $(run_case(run).name), $member, $(run.meta["level"]), Δt = $(run.meta["Δt"]) s", fontsize=22)

ax_a = Axis(fig[1, 1]; xlabel="t (s)", ylabel="uˢ(x_FOV, 0, t) (mm/s)", title="(a) Packet trajectory at the FOV")
lines!(ax_a, stats["t"], 1e3 .* stats["uˢ_fov"]; label="sampled", linewidth=3)
lines!(ax_a, stats["t"], 1e3 .* analytic; label="Uˢ₀ exp[−(t − t_peak)²/τ₀²]", linestyle=:dash, color=:black)
vlines!(ax_a, [τ, 3τ, 4τ, 5τ, 7τ]; color=(:gray, 0.5))
axislegend(ax_a; position=:lt)

ax_b = Axis(fig[1, 2]; xlabel="x (m)", ylabel="t (s)", title="(b) surface ⟨u^L⟩_y (mm/s)")
hm_b = heatmap!(ax_b, x, t, 1e3 .* UL[:, end, :]; colormap=:balance, colorrange=1e3 .* symmetric_range(UL[:, end, :]))
lines!(ax_b, [packet_center(tn, p) for tn in t], t; color=:black, linestyle=:dash)
vlines!(ax_b, [p.x_FOV]; color=:black)
Colorbar(fig[1, 3], hm_b)

ax_c = Axis(fig[2, 1]; xlabel="x (m)", ylabel="t (s)", title="(c) surface $label (mm/s)")
hm_c = heatmap!(ax_c, x, t, 1e3 .* UE[:, end, :]; colormap=:balance, colorrange=1e3 .* symmetric_range(UE[:, end, :]))
lines!(ax_c, [packet_center(tn, p) for tn in t], t; color=:black, linestyle=:dash)
vlines!(ax_c, [p.x_FOV]; color=:black)

ax_d = Axis(fig[2, 2]; xlabel="ξ = x − x_c(t) (m)", ylabel="t (s)", title="(d) surface $label in packet coordinates (mm/s)")
hm_d = heatmap!(ax_d, ξ, t, 1e3 .* Uξ; colormap=:balance, colorrange=1e3 .* symmetric_range(UE[:, end, :]))
vlines!(ax_d, [-p.σ₀, 0, p.σ₀]; color=:black, linestyle=:dot)
Colorbar(fig[2, 3], hm_d)

ax_e = Axis(fig[3, 1]; xlabel="⟨u^E⟩_y(x_FOV, z) (mm/s)", ylabel="k₀ z", title="(e) FOV profiles")
for (n_width, color) in zip((1, 3, 4, 5, 7, 8), Makie.wong_colors())
    n = nearest_index(t, n_width * τ)
    lines!(ax_e, 1e3 .* UE[i_fov, :, n], k .* z; label="t = $(n_width)τ₀", color, linewidth=2)
end
vlines!(ax_e, [0]; color=:black)
axislegend(ax_e; position=:rb)

ax_f = Axis(fig[3, 2]; xlabel="t (s)", ylabel="k₀ z", title="(f) $label at the FOV (mm/s)")
hm_f = heatmap!(ax_f, t, k .* z, 1e3 .* permutedims(UE[i_fov, :, :]); colormap=:balance,
                colorrange=1e3 .* symmetric_range(UE[i_fov, :, :]))
vlines!(ax_f, [τ, 3τ, 4τ, 5τ, 7τ]; color=(:black, 0.5), linestyle=:dot)
Colorbar(fig[3, 3], hm_f)

output = get(args, "output", joinpath(figure_directory(), "packet_hovmoller_$(member)_$(run.meta["level"]).png"))
save(output, fig)
@info "Saved $output"
