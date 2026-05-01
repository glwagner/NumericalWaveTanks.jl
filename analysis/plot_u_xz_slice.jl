using CairoMakie
using Oceananigans
using Printf

# Plot a side-view (xz) slice of u at four times, given a run output directory.
# Usage: julia --project analysis/plot_u_xz_slice.jl <run_dir>

run_dir = ARGS[1]
prefix  = basename(rstrip(run_dir, '/'))
xz_file = joinpath(run_dir, prefix * "_xz_left.jld2")

isfile(xz_file) || error("could not find $xz_file")

@info "Loading $(xz_file)"
u_ts = FieldTimeSeries(xz_file, "u")
times = u_ts.times
xs = xnodes(u_ts)
zs = znodes(u_ts)

# Pick four roughly evenly-spaced snapshots from start, 1/3, 2/3, end
nt = length(times)
idx = [1, max(1, div(nt, 3)), max(1, div(2nt, 3)), nt]

@info "Times available: $(length(times)) snapshots from $(first(times))s to $(last(times))s"

fig = Figure(size=(1600, 420), fontsize=14)

u_max_global = maximum(abs(maximum(interior(u_ts[i]))) for i in idx)
u_min_global = minimum(minimum(interior(u_ts[i])) for i in idx)
crange = (u_min_global, max(abs(u_min_global), abs(u_max_global)))

for (j, i) in enumerate(idx)
    ax = Axis(fig[1, j];
              xlabel = "x (m)",
              ylabel = j == 1 ? "z (m)" : "",
              title = @sprintf("t = %.2f s", times[i]),
              aspect = DataAspect())
    u_xz = interior(u_ts[i], :, 1, :)
    hm = heatmap!(ax, xs, zs, u_xz; colormap=:balance, colorrange=crange)
    if j == length(idx)
        Colorbar(fig[1, j+1], hm; label="u (m/s)")
    end
end

Label(fig[0, :], "Side-view (x–z) u-velocity slice — $prefix";
      fontsize = 16, halign = :center)

out = joinpath(run_dir, "u_xz_slice.png")
save(out, fig; px_per_unit=2)
@info "Saved $out"
