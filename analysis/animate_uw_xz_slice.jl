using CairoMakie
using Oceananigans
using Printf

# Animate u(x, z) and w(x, z) for three runs in a 3-row × 2-col grid:
#   row i, col 1 = u for run i;  row i, col 2 = w for run i
# Two colorbars (one per field), one set of common time axis.
# Usage: julia --project analysis/animate_uw_xz_slice.jl <run1> <run2> <run3> [out.mp4]

function main(run_dirs, out)
    @info "Loading u, w xz_left slices for $(length(run_dirs)) runs"
    labels  = String[]
    u_series = []
    w_series = []
    xs_u, zs_u, xs_w, zs_w = [], [], [], []
    times = nothing
    for d in run_dirs
        prefix  = basename(rstrip(d, '/'))
        f       = joinpath(d, prefix * "_xz_left.jld2")
        isfile(f) || error("missing $f")
        u_ts = FieldTimeSeries(f, "u")
        w_ts = FieldTimeSeries(f, "w")
        push!(labels, prefix)
        push!(u_series, u_ts)
        push!(w_series, w_ts)
        push!(xs_u, xnodes(u_ts)); push!(zs_u, znodes(u_ts))
        push!(xs_w, xnodes(w_ts)); push!(zs_w, znodes(w_ts))
        if times === nothing
            times = u_ts.times
        end
    end

    @info "Computing global colorranges"
    u_max = 0.0
    w_max = 0.0
    for (u_ts, w_ts) in zip(u_series, w_series), k in 1:length(u_ts.times)
        u_max = max(u_max, maximum(abs, interior(u_ts[k])))
        w_max = max(w_max, maximum(abs, interior(w_ts[k])))
    end
    u_crange = (-u_max, u_max)
    w_crange = (-w_max, w_max)
    @info @sprintf("u_crange = (%.3e, %.3e), w_crange = (%.3e, %.3e)",
                   u_crange..., w_crange...)

    nframes = minimum(length(u.times) for u in u_series)
    @info "Animating $nframes frames"

    fig = Figure(size=(1700, 1050), fontsize=14)
    time_obs = Observable(1)

    # Short title derived from the run prefix
    function shortlabel(prefix)
        m = match(r"N(\d+)_(\d+)_(\d+)", prefix)
        Nstr = m === nothing ? "?" : "$(m.captures[1])³"
        if startswith(prefix, "w23_les")
            return "LES baseline   ($Nstr)"
        elseif startswith(prefix, "stratified")
            return "Stratified   ($Nstr)"
        else
            return prefix
        end
    end

    # Detect whether a run is stratified and at what depth
    # The stratified script encodes zₕ in the prefix as e.g. "zh030" → -0.030 m, "zh002" → -0.002 m
    is_stratified(prefix) = startswith(prefix, "stratified")
    function thermocline_depth(prefix)
        m = match(r"zh(\d+)", prefix)
        m === nothing && return -0.03
        return -parse(Int, m.captures[1]) / 1000.0
    end

    nruns = length(u_series)
    local hm_u_ref, hm_w_ref
    for (i, (u_ts, w_ts)) in enumerate(zip(u_series, w_series))
        ax_u = Axis(fig[i, 1];
                    xlabel = i == nruns ? "x (m)" : "",
                    ylabel = "z (m)",
                    title  = shortlabel(labels[i]) * "  —  u",
                    aspect = DataAspect())
        ax_w = Axis(fig[i, 3];
                    xlabel = i == nruns ? "x (m)" : "",
                    ylabel = "",
                    title  = shortlabel(labels[i]) * "  —  w",
                    aspect = DataAspect())

        field_u = @lift interior(u_ts[$time_obs], :, 1, :)
        field_w = @lift interior(w_ts[$time_obs], :, 1, :)

        hm_u = heatmap!(ax_u, xs_u[i], zs_u[i], field_u; colormap=:balance, colorrange=u_crange)
        hm_w = heatmap!(ax_w, xs_w[i], zs_w[i], field_w; colormap=:balance, colorrange=w_crange)

        if is_stratified(labels[i])
            zh = thermocline_depth(labels[i])
            hlines!(ax_u, zh; color = :black, linestyle = :dash, linewidth = 1)
            hlines!(ax_w, zh; color = :black, linestyle = :dash, linewidth = 1)
        end

        if i == 1
            hm_u_ref = hm_u
            hm_w_ref = hm_w
        end
    end

    Colorbar(fig[1:nruns, 2], hm_u_ref; label="u (m/s)", width=18)
    Colorbar(fig[1:nruns, 4], hm_w_ref; label="w (m/s)", width=18)

    Label(fig[0, 1:4], @lift(@sprintf("t = %.2f s", times[$time_obs]));
          fontsize=20, halign=:center)

    rowgap!(fig.layout, 8)
    colgap!(fig.layout, 10)

    # Save a static last-frame for layout verification
    time_obs[] = nframes
    save(replace(out, ".mp4" => "_lastframe.png"), fig; px_per_unit=2)
    @info "Saved $(replace(out, ".mp4" => "_lastframe.png"))"

    @info "Recording $out"
    record(fig, out, 1:nframes; framerate=12) do n
        time_obs[] = n
    end

    @info "Saved $out"
end

# Last arg ending in .mp4 is the output; everything before is a run dir.
let args = ARGS
    out_idx = findlast(a -> endswith(a, ".mp4"), args)
    if out_idx === nothing
        main(args, "uw_xz_animation.mp4")
    else
        main(args[1:out_idx-1], args[out_idx])
    end
end
