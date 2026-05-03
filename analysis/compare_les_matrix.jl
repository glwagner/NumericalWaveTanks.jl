using CairoMakie
using Oceananigans
using Printf

# Side-by-side comparison of LES profile-time-series across runs.
# Usage: julia --project compare_les_matrix.jl <dir1> <dir2> <dir3> [out.png]

function load_profiles(dir)
    prefix = basename(rstrip(dir, '/'))
    f = joinpath(dir, prefix * "_profiles.jld2")
    isfile(f) || error("missing $f")
    U  = FieldTimeSeries(f, "U")
    uu = FieldTimeSeries(f, "uu")
    ww = FieldTimeSeries(f, "ww")
    uw = FieldTimeSeries(f, "uw")
    return (; U, uu, ww, uw, prefix)
end

function shortlabel(prefix)
    m = match(r"ep(\d+)", prefix)
    eps = m === nothing ? "?" : "$(parse(Int, m.captures[1])/1000)"
    return "ε = $eps"
end

function stack(ts; face=false)
    Nt = length(ts.times)
    z  = znodes(ts)
    Nz = length(z)
    arr = zeros(Float32, Nz, Nt)
    for n in 1:Nt
        arr[:, n] = vec(interior(ts[n]))
    end
    return ts.times, z, arr
end

function compare_runs(dirs, out)
    runs = [load_profiles(d) for d in dirs]
    nr = length(runs)
    @info "Loaded $nr runs"

    # Stack data for each run
    data = [(stack(r.U), stack(r.uu), stack(r.ww; face=true), stack(r.uw)) for r in runs]

    # Subtract U² from uu to get u'²
    for i in 1:nr
        Uts, Uz, Uar = data[i][1]
        uts, uz, uar = data[i][2]
        for n in 1:length(uts), k in 1:size(uar, 1)
            uar[k, n] = max(uar[k, n] - Uar[k, n]^2, 0f0)
        end
    end

    # Global colorranges per quantity
    Umax = maximum(maximum(d[1][3]) for d in data)
    upmax = maximum(maximum(d[2][3]) for d in data)
    wpmax = maximum(maximum(d[3][3]) for d in data)
    uwmax = maximum(maximum(abs, d[4][3]) for d in data)

    fig = Figure(size=(450 * nr + 100, 1300), fontsize=14)
    rowlabels = ["U(z, t) [m/s]", "⟨u'²⟩(z, t) [m²/s²]", "⟨w²⟩(z, t) [m²/s²]", "⟨u'w'⟩(z, t) [m²/s²]"]
    cmaps = (:thermal, :viridis, :viridis, :balance)

    local hms = Vector{Any}(undef, 4)

    for (i, r) in enumerate(runs)
        ts_U, z_U, U_arr = data[i][1]
        ts_u, z_u, up_arr = data[i][2]
        ts_w, z_w, ww_arr = data[i][3]
        ts_uw, z_uw, uw_arr = data[i][4]

        for (j, (data_arr, zs)) in enumerate(((U_arr, z_U), (up_arr, z_u), (ww_arr, z_w), (uw_arr, z_uw)))
            ax = Axis(fig[j, i];
                      xlabel = j == 4 ? "t (s)" : "",
                      ylabel = i == 1 ? "z (m)" : "",
                      title = j == 1 ? shortlabel(r.prefix) : "")
            cmap = cmaps[j]
            crange = j == 1 ? (0, Umax) :
                     j == 4 ? (-uwmax, uwmax) :
                     j == 2 ? (0, upmax) : (0, wpmax)
            hm = heatmap!(ax, ts_U, zs, data_arr'; colormap=cmap, colorrange=crange)
            i == nr && (hms[j] = hm)
        end
    end

    # Colorbars on the right
    for j in 1:4
        Colorbar(fig[j, nr + 1], hms[j]; label=rowlabels[j], width=18)
    end

    Label(fig[0, 1:nr+1], "LES matrix: U, ⟨u'²⟩, ⟨w²⟩, ⟨u'w'⟩ vs ε   (DNS-IC at t=19 s, run to t=60 s)";
          fontsize=16)

    save(out, fig; px_per_unit=2)
    @info "Saved $out"
end

if abspath(PROGRAM_FILE) == @__FILE__
    dirs = ARGS[1:end-1]
    out = ARGS[end]
    if !endswith(out, ".png")
        dirs = ARGS
        out = "les_matrix_compare.png"
    end
    compare_runs(dirs, out)
end
