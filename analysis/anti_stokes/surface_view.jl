#####
##### Top-down view of the surface for several runs at once (xy_surface.jld2, written with
##### animation=true): vertical velocity w one cell below the surface and the streamwise anomaly
##### u′ = u − ⟨u⟩_y in the top cell, as a snapshot and as an animation over a common time window.
##### Streamwise-elongated streaks of u′ with alternating bands of w are the Langmuir-cell signature.
#####
##### Usage: julia --project=. analysis/anti_stokes/surface_view.jl runs=<dir1>,<dir2>,... labels="a|b|..."
#####            [snapshot=14 tmax=22.4 stride=2 framerate=12 name=surface_view]
#####

using CairoMakie
include("common.jl")

args = parse_key_value_args(ARGS)
dirs = String.(split(args["runs"], ','))
labels = String.(split(get(args, "labels", join(basename.(dirname.(dirs)), "|")), '|'))
t_snap = getarg(args, "snapshot", 14.0)
t_max = getarg(args, "tmax", 22.4)
stride = getarg(args, "stride", 2)
framerate = getarg(args, "framerate", 12)
name = getarg(args, "name", "surface_view")
length(labels) == length(dirs) || error("one label per run")

@info "Loading surface slices..."
W = [FieldTimeSeries(joinpath(d, "xy_surface.jld2"), "w") for d in dirs]
U = [FieldTimeSeries(joinpath(d, "xy_surface.jld2"), "u") for d in dirs]
t = collect(Float64, W[1].times)
for w in W[2:end]
    tw = collect(Float64, w.times)
    length(tw) >= length(findall(<=(t_max + 1e-6), t)) || error("runs have different output cadences")
end
nmax = findlast(<=(t_max + 1e-6), t)
grid = W[1].grid
x = collect(Float64, Array(xnodes(grid, Center())))
xf = collect(Float64, Array(xnodes(grid, Face())))
y = collect(Float64, Array(ynodes(grid, Center())))
Lx, Ly = grid.Lx, grid.Ly

wfield(i, n) = Float64.(Array(interior(W[i][n]))[:, :, 1])
function ufield(i, n)
    u = Float64.(Array(interior(U[i][n]))[:, :, 1])
    return u .- mean(u; dims=2)          # remove the y-mean (packet / mean current / Stokes drift)
end
n_snap = nearest_index(t, t_snap)
wmax = maximum(quantile(abs.(vec(wfield(i, n_snap))), 0.995) for i in eachindex(dirs))
umax = maximum(quantile(abs.(vec(ufield(i, n_snap))), 0.995) for i in eachindex(dirs))

function draw!(fig, n)
    for (i, lbl) in enumerate(labels)
        axw = Axis(fig[i, 1]; title = i == 1 ? "vertical velocity w (mm/s), one cell below the surface" : "", ylabel = lbl, xlabel = i == length(dirs) ? "x (m)" : "", aspect = DataAspect(), titlesize = 16)
        heatmap!(axw, x, y, 1e3 .* wfield(i, n); colormap = :balance, colorrange = (-1e3wmax, 1e3wmax))
        axu = Axis(fig[i, 2]; title = i == 1 ? "streamwise anomaly u′ = u − ⟨u⟩_y (mm/s), top cell" : "", xlabel = i == length(dirs) ? "x (m)" : "", aspect = DataAspect(), titlesize = 16)
        heatmap!(axu, xf[1:size(ufield(i, n), 1)], y, 1e3 .* ufield(i, n); colormap = :balance, colorrange = (-1e3umax, 1e3umax))
        i < length(dirs) && (hidexdecorations!(axw; grid=false); hidexdecorations!(axu; grid=false))
        hideydecorations!(axu; grid=false)
    end
    Colorbar(fig[1:length(dirs), 3]; colormap = :balance, colorrange = (-1e3wmax, 1e3wmax), label = "w (mm/s)")
    Colorbar(fig[1:length(dirs), 4]; colormap = :balance, colorrange = (-1e3umax, 1e3umax), label = "u′ (mm/s)")
end

set_theme!(Theme(fontsize=16))
fig = Figure(size = (2400, 220 * length(dirs) + 120))
title = Observable(@sprintf("Surface view, t = %.1f s", t[n_snap]))
Label(fig[0, 1:4], title; fontsize = 20)
draw!(fig, n_snap)
snapshot = joinpath(figure_directory(), "$(name)_snapshot_t$(round(Int, t[n_snap]))s.png")
save(snapshot, fig)
@info "Saved $snapshot"

# animation: redraw the heatmaps through observables
fig = Figure(size = (2400, 220 * length(dirs) + 120))
n_obs = Observable(1)
Label(fig[0, 1:4], @lift(@sprintf("Surface view, t = %.1f s", t[$n_obs])); fontsize = 20)
for (i, lbl) in enumerate(labels)
    axw = Axis(fig[i, 1]; title = i == 1 ? "vertical velocity w (mm/s), one cell below the surface" : "", ylabel = lbl, xlabel = i == length(dirs) ? "x (m)" : "", aspect = DataAspect(), titlesize = 16)
    heatmap!(axw, x, y, @lift(1e3 .* wfield(i, $n_obs)); colormap = :balance, colorrange = (-1e3wmax, 1e3wmax))
    axu = Axis(fig[i, 2]; title = i == 1 ? "streamwise anomaly u′ = u − ⟨u⟩_y (mm/s), top cell" : "", xlabel = i == length(dirs) ? "x (m)" : "", aspect = DataAspect(), titlesize = 16)
    heatmap!(axu, xf[1:size(ufield(i, 1), 1)], y, @lift(1e3 .* ufield(i, $n_obs)); colormap = :balance, colorrange = (-1e3umax, 1e3umax))
    i < length(dirs) && (hidexdecorations!(axw; grid=false); hidexdecorations!(axu; grid=false))
    hideydecorations!(axu; grid=false)
end
Colorbar(fig[1:length(dirs), 3]; colormap = :balance, colorrange = (-1e3wmax, 1e3wmax), label = "w (mm/s)")
Colorbar(fig[1:length(dirs), 4]; colormap = :balance, colorrange = (-1e3umax, 1e3umax), label = "u′ (mm/s)")
output = joinpath(figure_directory(), "$(name).mp4")
frames = 1:stride:nmax
@info "Recording $(length(frames)) frames to $output"
CairoMakie.Makie.record(fig, output, frames; framerate) do n
    n_obs[] = n
end
@info "Saved $output"
