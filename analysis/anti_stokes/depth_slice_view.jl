#####
##### Horizontal slices at depth from the sparse 3D snapshots (snapshots.jld2) for several runs:
##### w and the streamwise anomaly u′ = u − ⟨u⟩_y at z = −depth, in an x window. Langmuir cells
##### show as streamwise bands of alternating w a few Stokes depths wide.
#####
##### Usage: julia --project=. analysis/anti_stokes/depth_slice_view.jl runs=<d1>,<d2>,... labels="a|b|..." time=14 depth=0.03 [xrange=3,8 name=depth_slice]
#####

using CairoMakie
include("common.jl")

args = parse_key_value_args(ARGS)
dirs = String.(split(args["runs"], ','))
labels = String.(split(get(args, "labels", join(basename.(dirname.(dirs)), "|")), '|'))
t_want = getarg(args, "time", 14.0)
depth = getarg(args, "depth", 0.03)
xr = parse.(Float64, split(getarg(args, "xrange", "3,8"), ','))
name = getarg(args, "name", "depth_slice")
scale = getarg(args, "scale", 1.0)     # px_per_unit of the saved PNG (0.5 halves the file for web pages)

fields = []
for d in dirs
    f = jldopen(joinpath(d, "snapshots.jld2"))
    its = sort(parse.(Int, keys(f["timeseries/t"])))
    ts = [f["timeseries/t/$i"] for i in its]
    n = argmin(abs.(ts .- t_want))
    w = f["timeseries/w/$(its[n])"]; u = f["timeseries/u/$(its[n])"]
    close(f)
    run = load_run(d; fields=("U",))
    grid = run_grid(run)
    zc = collect(Float64, Array(znodes(grid, Center()))); zf = collect(Float64, Array(znodes(grid, Face())))
    kc, kf = argmin(abs.(zc .+ depth)), argmin(abs.(zf .+ depth))
    x = collect(Float64, Array(xnodes(grid, Center()))); xf = collect(Float64, Array(xnodes(grid, Face())))
    y = collect(Float64, Array(ynodes(grid, Center())))
    ix = findall(xx -> xr[1] <= xx <= xr[2], x); ixf = findall(xx -> xr[1] <= xx <= xr[2], xf[1:size(u, 1)])
    wsl = Float64.(w[ix, :, kf]); usl = Float64.(u[ixf, :, kc]); usl .-= mean(usl; dims=2)
    push!(fields, (; x = x[ix], xf = xf[ixf], y, w = wsl, u = usl, t = ts[n], z = zf[kf]))
    @info @sprintf("%s: snapshot at t = %.1f s, z = %.3f m, w_rms = %.2f mm/s", basename(dirname(d)), ts[n], zf[kf], 1e3 * sqrt(mean(abs2, wsl)))
end
wmax = maximum(quantile(abs.(vec(f.w)), 0.995) for f in fields)
umax = maximum(quantile(abs.(vec(f.u)), 0.995) for f in fields)
Ly = fields[1].y[end] + (fields[1].y[2] - fields[1].y[1]) / 2
row_h = round(Int, 1000 * (Ly / (xr[2] - xr[1])) + 40)

set_theme!(Theme(fontsize=16))
fig = Figure(size = (2400, row_h * length(dirs) + 120))
Label(fig[0, 1:4], @sprintf("Horizontal slice at z = %.0f cm, x ∈ [%.0f, %.0f] m, t ≈ %.1f s", -100fields[1].z, xr..., fields[1].t); fontsize = 20)
for (i, (f, lbl)) in enumerate(zip(fields, labels))
    axw = Axis(fig[i, 1]; title = i == 1 ? "vertical velocity w (mm/s)" : "", ylabel = lbl, xlabel = i == length(dirs) ? "x (m)" : "", aspect = DataAspect(), titlesize = 16)
    heatmap!(axw, f.x, f.y, 1e3 .* f.w; colormap = :balance, colorrange = (-1e3wmax, 1e3wmax))
    axu = Axis(fig[i, 2]; title = i == 1 ? "streamwise anomaly u′ = u − ⟨u⟩_y (mm/s)" : "", xlabel = i == length(dirs) ? "x (m)" : "", aspect = DataAspect(), titlesize = 16)
    heatmap!(axu, f.xf, f.y, 1e3 .* f.u; colormap = :balance, colorrange = (-1e3umax, 1e3umax))
    i < length(dirs) && (hidexdecorations!(axw; grid=false); hidexdecorations!(axu; grid=false))
    hideydecorations!(axu; grid=false)
end
Colorbar(fig[1:length(dirs), 3]; colormap = :balance, colorrange = (-1e3wmax, 1e3wmax), label = "w (mm/s)")
Colorbar(fig[1:length(dirs), 4]; colormap = :balance, colorrange = (-1e3umax, 1e3umax), label = "u′ (mm/s)")
output = joinpath(figure_directory(), @sprintf("%s_z%.0fcm_t%.0fs.png", name, 100depth, fields[1].t))
save(output, fig; px_per_unit=scale)
@info "Saved $output"
