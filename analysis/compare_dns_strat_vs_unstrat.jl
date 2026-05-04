using CairoMakie
using Oceananigans
using Printf

# Side-by-side comparison: unstratified vs stratified DNS at the same t.
# Plots u(x, y, z=top) and u(x, z, y=0) and (for stratified) b(x, z, y=0).

const UNSTRAT = "constant_waves_ic005000_ep110_k30_alpha120_N768_768_512_L10_10_5"
const STRAT   = "strat_dns_ic005000_ep110_zh030_dr035_qb0_alpha120_N768_768_512_L10_10_5"

function file3d(dir)
    prefix = basename(rstrip(dir, '/'))
    return joinpath(dir, prefix * "_3d_fields.jld2")
end

function fileXZ(dir)
    prefix = basename(rstrip(dir, '/'))
    return joinpath(dir, prefix * "_xz_left.jld2")
end

function plot_compare(out)
    u_un_xy = FieldTimeSeries(file3d(UNSTRAT), "u")
    u_st_xy = FieldTimeSeries(file3d(STRAT),   "u")
    u_un_xz = FieldTimeSeries(fileXZ(UNSTRAT), "u")
    u_st_xz = FieldTimeSeries(fileXZ(STRAT),   "u")
    b_st_xz = FieldTimeSeries(fileXZ(STRAT),   "b")

    times = u_un_xy.times  # 1 s spacing for 3d_fields
    xs = xnodes(u_un_xy); ys = ynodes(u_un_xy); zs = znodes(u_un_xy)

    # Pick three times: 18, 20, 22
    target_t = [18.0, 20.0, 22.0]
    idx_3d   = [argmin(abs.(times .- t)) for t in target_t]

    times_xz = u_un_xz.times
    idx_xz = [argmin(abs.(times_xz .- t)) for t in target_t]

    fig = Figure(size=(1500, 1100), fontsize=13)

    # Color ranges
    u_max = 0.0
    for ts in (u_un_xy, u_st_xy), i in idx_3d
        u_max = max(u_max, maximum(abs, interior(ts[i], :, :, length(zs))))
    end
    crange_u = (-u_max, u_max)
    crange_uxz = (-0.25, 0.25)
    crange_b = (-0.343, 0.0)

    nt = length(target_t)
    local hms = Dict{String, Any}()

    # Row 1: u(x, y, z=top), unstratified
    for (j, i) in enumerate(idx_3d)
        ax = Axis(fig[1, j]; title=@sprintf("t = %.1f s — unstrat surface u", times[i]),
                  xlabel = j == 1 ? "x (m)" : "", ylabel = j == 1 ? "y (m)" : "",
                  aspect = DataAspect())
        u_top = interior(u_un_xy[i], :, :, length(zs))
        hm = heatmap!(ax, xs, ys, u_top; colormap=:balance, colorrange=crange_u)
        j == nt && (hms["u_un"] = hm)
    end
    Colorbar(fig[1, nt+1], hms["u_un"]; label="u (m/s)", width=15)

    # Row 2: u(x, y, z=top), stratified
    for (j, i) in enumerate(idx_3d)
        ax = Axis(fig[2, j]; title=@sprintf("t = %.1f s — strat zₕ=-3cm surface u", times[i]),
                  xlabel = j == 1 ? "x (m)" : "", ylabel = j == 1 ? "y (m)" : "",
                  aspect = DataAspect())
        u_top = interior(u_st_xy[i], :, :, length(zs))
        hm = heatmap!(ax, xs, ys, u_top; colormap=:balance, colorrange=crange_u)
    end

    # Row 3: stratified b(x, z) — shows if thermocline gets disturbed
    xs_xz = xnodes(b_st_xz); zs_xz = znodes(b_st_xz)
    for (j, i) in enumerate(idx_xz)
        ax = Axis(fig[3, j]; title=@sprintf("t = %.1f s — strat buoyancy", times_xz[i]),
                  xlabel = "x (m)", ylabel = j == 1 ? "z (m)" : "",
                  aspect = DataAspect())
        b_xz = interior(b_st_xz[i], :, 1, :)
        hm = heatmap!(ax, xs_xz, zs_xz, b_xz; colormap=:balance, colorrange=crange_b)
        hlines!(ax, [-0.03]; color=:black, linestyle=:dash, linewidth=1)
        j == nt && (hms["b"] = hm)
    end
    Colorbar(fig[3, nt+1], hms["b"]; label="b (m/s²)", width=15)

    Label(fig[0, 1:nt+1], "DNS: ε=0.11, U'=5cm/s, 0.1×0.1×0.05 m, 768²×512  —  stratified zₕ=-3 cm"; fontsize=15)

    save(out, fig; px_per_unit=2)
    @info "Saved $out"
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_compare(length(ARGS) >= 1 ? ARGS[1] : "dns_compare_strat_vs_unstrat.png")
end
