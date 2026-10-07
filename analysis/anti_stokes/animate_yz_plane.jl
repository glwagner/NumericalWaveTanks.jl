#####
##### Animation of the y-z plane at x_FOV (the virtual PIV plane, file fov_plane.jld2) for one run
##### and, optionally, its matched control: vertical velocity w and the Eulerian streamwise
##### velocity uᴱ = u − uˢ over the top part of the water column. Streamwise-oriented Langmuir
##### rolls appear as spanwise-alternating w with converging u anomalies.
#####
##### Usage: julia --project=. analysis/anti_stokes/animate_yz_plane.jl run=<dir> [control=<dir>]
#####            [output=<mp4> framerate=15 stride=1 depth=0.2]
#####

using CairoMakie
include("common.jl")

args = parse_key_value_args(ARGS)
run_dir = args["run"]
control_dir = get(args, "control", "")
framerate = getarg(args, "framerate", 15)
stride = getarg(args, "stride", 1)
depth = getarg(args, "depth", 0.2)
stills = haskey(args, "stills") ? parse.(Float64, split(args["stills"], ',')) : Float64[]

# suffix of the run directory beyond the standard seed/dt part (e.g. "_ustar4.5_long"), so movies of tagged runs get distinct names
run_tag(d) = (m = match(r"dt\d\.\d{3}(.*)$", basename(d)); isnothing(m) ? "" : m.captures[1])
run = load_run(run_dir; fields=("U",))
c = run_case(run)
τ, tp, k = τ₀(run), t_peak(run), k₀(run)
uniform = run_uniform(run)
stokes(z, t) = uniform ? uniform_uˢ(z, t, run_uniform_parameters(run)) :
               (run.meta["has_packet"] ? uˢ(run.meta["x_FOV"], 0, z, t, run_packet(run)) : 0.0)

@info "Loading the y-z plane..."
w_ts = FieldTimeSeries(joinpath(run_dir, "fov_plane.jld2"), "w")
u_ts = FieldTimeSeries(joinpath(run_dir, "fov_plane.jld2"), "u")
has_control = !isempty(control_dir)
w_ct = has_control ? FieldTimeSeries(joinpath(control_dir, "fov_plane.jld2"), "w") : nothing
t = collect(Float64, w_ts.times)
Nt = length(t)
grid = w_ts.grid
y = collect(Float64, Array(ynodes(grid, Center())))
zc = collect(Float64, Array(znodes(grid, Center())))
zf = collect(Float64, Array(znodes(grid, Face())))
kc = findall(z -> z >= -depth, zc); kf = findall(z -> z >= -depth, zf)

wmax = quantile(abs.(vec(Array(interior(w_ts[nearest_index(t, tp)]))[1, :, kf])), 0.995)
umax = quantile(abs.(vec(Array(interior(u_ts[nearest_index(t, tp + 2τ)]))[1, :, kc]) .- stokes.(zc[kc]', tp + 2τ) |> vec), 0.995)
wmax = max(wmax, 1e-4); umax = max(umax, 1e-3)

frames = 1:stride:Nt
n_obs = Observable(1)
w_obs = @lift Array(interior(w_ts[$n_obs]))[1, :, kf]
uE_obs = @lift Array(interior(u_ts[$n_obs]))[1, :, kc] .- stokes.(zc[kc]', t[$n_obs])
title_obs = @lift @sprintf("%s, %s seed %d: t = %.1f s, age (t − t_peak)/τ₀ = %+.2f, uˢ(0, t)/Uˢ₀ = %.2f",
                           run.meta["member"], run.meta["level"], run.meta["seed"], t[$n_obs], (t[$n_obs] - tp) / τ,
                           stokes(0.0, t[$n_obs]) / Float64(c.Uˢ₀))

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1800, has_control ? 1250 : 850))
Label(fig[0, 1:2], title_obs, fontsize=20)
ax_w = Axis(fig[1, 1]; xlabel="y (m)", ylabel="z (m)", title="vertical velocity w (mm/s) in the y-z plane at x_FOV", aspect=DataAspect())
hm_w = heatmap!(ax_w, y, zf[kf], @lift(1e3 .* $w_obs); colormap=:balance, colorrange=(-1e3wmax, 1e3wmax))
Colorbar(fig[1, 2], hm_w; label="w (mm/s)")
ax_u = Axis(fig[2, 1]; xlabel="y (m)", ylabel="z (m)", title="Eulerian streamwise velocity uᴱ = u − uˢ(z, t) (mm/s)", aspect=DataAspect())
hm_u = heatmap!(ax_u, y, zc[kc], @lift(1e3 .* $uE_obs); colormap=:balance, colorrange=(-1e3umax, 1e3umax))
Colorbar(fig[2, 2], hm_u; label="uᴱ (mm/s)")
if has_control
    wc_obs = @lift Array(interior(w_ct[$n_obs]))[1, :, kf]
    ax_c = Axis(fig[3, 1]; xlabel="y (m)", ylabel="z (m)", title="control (same turbulence, no waves): w (mm/s)", aspect=DataAspect())
    hm_c = heatmap!(ax_c, y, zf[kf], @lift(1e3 .* $wc_obs); colormap=:balance, colorrange=(-1e3wmax, 1e3wmax))
    Colorbar(fig[3, 2], hm_c; label="w (mm/s)")
end

output = get(args, "output", joinpath(figure_directory(),
             "yz_plane_$(run.meta["member"])_$(case_dirname(c))_$(run.meta["level"])_seed$(run.meta["seed"])$(run_tag(run_dir)).mp4"))
@info "Recording $(length(frames)) frames to $output"
CairoMakie.Makie.record(fig, output, frames; framerate) do frame
    n_obs[] = frame
end
n_obs[] = nearest_index(t, tp + τ)
still = replace(output, ".mp4" => "_age1.png")
save(still, fig)
for ts in stills
    n_obs[] = nearest_index(t, ts)
    s = replace(output, ".mp4" => @sprintf("_t%.0fs.png", ts))
    save(s, fig); @info "Saved $s"
end
@info "Saved $output and $still"
