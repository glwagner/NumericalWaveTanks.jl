#####
##### Animation of the turbulent field as the packet passes, from the high-cadence slices
##### written with `animation=true` (files xy_surface.jld2 and xz_slice.jld2):
#####
#####   (1) surface vertical velocity w, packet + turbulence member
#####   (2) surface vertical velocity w, matched no-wave control (same turbulence)
#####   (3) packet-induced perturbation of surface w: (1) − (2)
#####   (4) x-z slice of the Eulerian streamwise perturbation u^E − u_control at y = Ly/2,
#####       with the prescribed Stokes envelope uˢ(x, 0, t) drawn above
#####
##### Usage: julia --project=. analysis/anti_stokes/animate_packet_turbulence.jl packet=<dir> control=<dir>
#####            [output=<mp4> framerate=15 stride=1]
#####

using CairoMakie
include("common.jl")

args = parse_key_value_args(ARGS)
packet_dir, control_dir = args["packet"], args["control"]
framerate = getarg(args, "framerate", 15)
stride = getarg(args, "stride", 1)

pk = load_run(packet_dir; fields=("U",))
p, c = run_packet(pk), run_case(pk)
τ, tp, k = τ₀(pk), t_peak(pk), k₀(pk)
Lx, Ly = pk.meta["Lx"], pk.meta["Ly"]

@info "Loading slices..."
w_top_pk = FieldTimeSeries(joinpath(packet_dir, "xy_surface.jld2"), "w")
w_top_ct = FieldTimeSeries(joinpath(control_dir, "xy_surface.jld2"), "w")
u_xz_pk  = FieldTimeSeries(joinpath(packet_dir, "xz_slice.jld2"), "u")
u_xz_ct  = FieldTimeSeries(joinpath(control_dir, "xz_slice.jld2"), "u")

t = collect(Float64, w_top_pk.times)
length(t) == length(w_top_ct.times) || error("Packet and control slices have different lengths")
Nt = length(t)

grid = w_top_pk.grid
xc = collect(Float64, Array(xnodes(grid, Center())))
xf = collect(Float64, Array(xnodes(grid, Face())))
yc = collect(Float64, Array(ynodes(grid, Center())))
zc = collect(Float64, Array(znodes(grid, Center())))
z_w = Float64(Array(znodes(grid, Face()))[grid.Nz])   # height of the saved w plane

W_pk = Array{Float64}(Array(interior(w_top_pk))[:, :, 1, :])   # (Nx, Ny, Nt)
W_ct = Array{Float64}(Array(interior(w_top_ct))[:, :, 1, :])
U_pk = Array{Float64}(Array(interior(u_xz_pk))[:, 1, :, :])    # (Nx, Nz, Nt)
U_ct = Array{Float64}(Array(interior(u_xz_ct))[:, 1, :, :])

# Eulerian streamwise perturbation in the x-z slice: (u^L − uˢ) − u_control
ΔU = similar(U_pk)
for n in 1:Nt, kk in eachindex(zc), i in eachindex(xf)
    ΔU[i, kk, n] = U_pk[i, kk, n] - uˢ(xf[i], 0, zc[kk], t[n], p) - U_ct[i, kk, n]
end
ΔW = W_pk .- W_ct

w_lim = 1e3 * quantile(abs.(vec(W_ct)), 0.995)
dw_lim = 1e3 * max(quantile(abs.(vec(ΔW)), 0.995), 1e-6)
du_lim = 1e3 * max(quantile(abs.(vec(ΔU)), 0.995), 1e-6)

set_theme!(Theme(fontsize=20))
fig = Figure(size=(2200, 1500))
n = Observable(1)

title = @lift @sprintf("Case %s, %s, seed %d — t = %5.2f s = %.2f τ₀, packet centre x_c = %5.2f m",
                        c.name, pk.meta["level"], pk.meta["seed"], t[$n], t[$n] / τ, packet_center(t[$n], p))
Label(fig[0, 1:2], title; fontsize=26)

# Stokes envelope at the surface
ax0 = Axis(fig[1, 1]; ylabel="uˢ (mm/s)", title="prescribed surface Stokes drift", height=120,
           xticklabelsvisible=false)
env = @lift [1e3 * uˢ(x, 0, 0, t[$n], p) for x in xc]
lines!(ax0, xc, env; color=:black, linewidth=3)
vlines!(ax0, [p.x_FOV]; color=:red, linestyle=:dash)
ylims!(ax0, -2, 1e3 * p.Uˢ₀ * 1.1)
xlims!(ax0, 0, Lx)

ax1 = Axis(fig[2, 1]; ylabel="y (m)", title=@sprintf("(1) w at z = %.1f mm, packet + turbulence (mm/s)", 1e3 * z_w),
           aspect=DataAspect(), xticklabelsvisible=false)
hm1 = heatmap!(ax1, xc, yc, @lift(1e3 .* W_pk[:, :, $n]); colormap=:balance, colorrange=(-w_lim, w_lim))
vlines!(ax1, [p.x_FOV]; color=:red, linestyle=:dash)
Colorbar(fig[2, 2], hm1)

ax2 = Axis(fig[3, 1]; ylabel="y (m)", title="(2) w, no-wave control with the same turbulence (mm/s)",
           aspect=DataAspect(), xticklabelsvisible=false)
hm2 = heatmap!(ax2, xc, yc, @lift(1e3 .* W_ct[:, :, $n]); colormap=:balance, colorrange=(-w_lim, w_lim))
vlines!(ax2, [p.x_FOV]; color=:red, linestyle=:dash)
Colorbar(fig[3, 2], hm2)

ax3 = Axis(fig[4, 1]; ylabel="y (m)", title="(3) packet-induced perturbation of w: (1) − (2) (mm/s)",
           aspect=DataAspect(), xticklabelsvisible=false)
hm3 = heatmap!(ax3, xc, yc, @lift(1e3 .* ΔW[:, :, $n]); colormap=:balance, colorrange=(-dw_lim, dw_lim))
vlines!(ax3, [p.x_FOV]; color=:red, linestyle=:dash)
xc_line = @lift [packet_center(t[$n], p)]
vlines!(ax3, xc_line; color=:black)
Colorbar(fig[4, 2], hm3)

ax4 = Axis(fig[5, 1]; xlabel="x (m)", ylabel="z (m)",
           title="(4) Eulerian streamwise perturbation u^E − u_control at y = Ly/2 (mm/s)")
hm4 = heatmap!(ax4, xf, zc, @lift(1e3 .* ΔU[:, :, $n]); colormap=:balance, colorrange=(-du_lim, du_lim))
vlines!(ax4, [p.x_FOV]; color=:red, linestyle=:dash)
vlines!(ax4, xc_line; color=:black)
hlines!(ax4, [-Float64(c.δˢ)]; color=(:black, 0.5), linestyle=:dot)
ylims!(ax4, -Float64(c.h), 0)
xlims!(ax4, 0, Lx)
Colorbar(fig[5, 2], hm4)

rowsize!(fig.layout, 5, Relative(0.28))

output = get(args, "output", joinpath(figure_directory(),
             "packet_turbulence_animation_$(case_dirname(c))_$(pk.meta["level"])$(is_bounded_x(pk) ? "_boundedx" : "")_seed$(pk.meta["seed"]).mp4"))
frames = 1:stride:Nt
@info "Recording $(length(frames)) frames to $output"
CairoMakie.Makie.record(fig, output, frames; framerate) do frame
    n[] = frame
end
@info "Saved $output"

# Also save a still at the packet peak
n[] = nearest_index(t, tp)
still = replace(output, ".mp4" => "_peak.png")
save(still, fig)
@info "Saved $still"
