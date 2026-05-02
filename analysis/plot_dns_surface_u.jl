using CairoMakie
using Oceananigans
using JLD2
using Printf

# Plot u(x, y, z=top) at several times from a DNS 3D-fields file.
# This is the natural Langmuir-instability signature (along-wind streaks).
# Usage:
#   julia --project analysis/plot_dns_surface_u.jl <run_dir>

run_dir = ARGS[1]
prefix  = basename(rstrip(run_dir, '/'))
fields_file = joinpath(run_dir, prefix * "_3d_fields.jld2")
isfile(fields_file) || error("missing $fields_file")

@info "Loading u from $fields_file"
u_ts = FieldTimeSeries(fields_file, "u")
times = u_ts.times
xs = xnodes(u_ts)
ys = ynodes(u_ts)
zs = znodes(u_ts)
@info "snapshots: $(length(times)), times: $times"
@info "z(top) = $(zs[end])"

# Pick 4 snapshots spread across the time range. Skip the first if it
# corresponds to the very first save (which holds the unphysical IC noise);
# use 4 evenly-spaced post-IC snapshots instead.
nt = length(times)
if nt >= 5
    idx = round.(Int, range(2, nt, length=4))
else
    idx = collect(1:nt)
end
idx = unique(idx)

# Compute global symmetric color range over the chosen frames at the top z
u_max = let m = 0.0
    for i in idx
        u_top = interior(u_ts[i], :, :, length(zs))
        m = max(m, maximum(abs, u_top))
    end
    m
end
crange = (-u_max, u_max)
@info @sprintf("u colorrange: ±%.3e m/s", u_max)

ncols = length(idx)
fig = Figure(size=(360 * ncols + 100, 380), fontsize=13)
hm_ref = let h = nothing
    for (j, i) in enumerate(idx)
        ax = Axis(fig[1, j];
                  xlabel = "x (m)",
                  ylabel = j == 1 ? "y (m)" : "",
                  title = @sprintf("t = %.2f s", times[i]),
                  aspect = DataAspect())
        u_top = interior(u_ts[i], :, :, length(zs))
        hm = heatmap!(ax, xs, ys, u_top; colormap=:balance, colorrange=crange)
        j == 1 && (h = hm)
    end
    h
end
Colorbar(fig[1, ncols + 1], hm_ref; label = "u(x, y, z=0) [m/s]", width = 18)
Label(fig[0, 1:ncols+1], "Surface u (xy at z = top) — $prefix"; fontsize = 15)

out = joinpath(run_dir, "surface_u_xy.png")
save(out, fig; px_per_unit = 2)
@info "Saved $out"
