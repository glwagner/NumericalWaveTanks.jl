using CairoMakie
using Oceananigans
using Printf

# Animate u(x, z) as a 3-row stack (one row per run), driven by time.
# Usage: julia --project analysis/animate_u_xz_slice.jl <run_dir1> <run_dir2> <run_dir3> [output.mp4]

function main(run_dirs, out)
    @info "Loading u xz_left slices for $(length(run_dirs)) runs"
    labels   = String[]
    u_series = []
    xs_all   = []
    zs_all   = []
    times    = nothing
    for d in run_dirs
        prefix  = basename(rstrip(d, '/'))
        f       = joinpath(d, prefix * "_xz_left.jld2")
        isfile(f) || error("missing $f")
        u_ts = FieldTimeSeries(f, "u")
        push!(labels, prefix)
        push!(u_series, u_ts)
        push!(xs_all, xnodes(u_ts))
        push!(zs_all, znodes(u_ts))
        if times === nothing
            times = u_ts.times
        end
    end

    @info "Computing global colorrange"
    all_max = 0.0
    for u_ts in u_series, k in 1:length(u_ts.times)
        all_max = max(all_max, maximum(abs, interior(u_ts[k])))
    end
    crange = (-all_max, all_max)
    @info @sprintf("crange = (%.3e, %.3e)", crange...)

    nframes = minimum(length(u.times) for u in u_series)
    @info "Animating $nframes frames"

    fig = Figure(size=(1100, 900), fontsize=14)
    time_obs = Observable(1)

    local hm_ref
    for (i, u_ts) in enumerate(u_series)
        ax = Axis(fig[i, 1];
                  xlabel = i == length(u_series) ? "x (m)" : "",
                  ylabel = "z (m)",
                  title  = labels[i],
                  aspect = DataAspect())
        field = @lift interior(u_ts[$time_obs], :, 1, :)
        hm = heatmap!(ax, xs_all[i], zs_all[i], field; colormap=:balance, colorrange=crange)
        if i == 1
            hm_ref = hm
        end
        hidexdecorations!(ax; ticks=false, label = i != length(u_series))
    end

    Colorbar(fig[1:length(u_series), 2], hm_ref;
             label="u (m/s)", width=18)

    Label(fig[0, 1:2], @lift(@sprintf("t = %.2f s", times[$time_obs]));
          fontsize=18, halign=:center)

    rowgap!(fig.layout, 8)
    colgap!(fig.layout, 10)

    # Save a single static frame first so we can verify layout cheaply
    time_obs[] = nframes
    save(replace(out, ".mp4" => "_lastframe.png"), fig; px_per_unit=2)
    @info "Saved $(replace(out, ".mp4" => "_lastframe.png")) for layout check"

    @info "Recording $out"
    record(fig, out, 1:nframes; framerate=12) do n
        time_obs[] = n
    end

    @info "Saved $out"
end

main(ARGS[1:3], length(ARGS) >= 4 ? ARGS[4] : "u_xz_animation.mp4")
