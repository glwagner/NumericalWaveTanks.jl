using CairoMakie
using Oceananigans
using Printf

# Side-by-side comparison of STRATIFIED LES profile time series.
# Same structure as compare_les_matrix.jl but plots U, B, <w²>, <w'b'>.
# Usage: julia --project compare_strat_les_matrix.jl <dir1> <dir2> ... <out.png>

function load_profiles(dir)
    prefix = basename(rstrip(dir, '/'))
    f = joinpath(dir, prefix * "_profiles.jld2")
    isfile(f) || error("missing $f")
    U  = FieldTimeSeries(f, "U")
    B  = FieldTimeSeries(f, "B")
    ww = FieldTimeSeries(f, "ww")
    wb = FieldTimeSeries(f, "wb")
    return (; U, B, ww, wb, prefix)
end

function shortlabel(prefix)
    m_zh = match(r"zh(\d+)", prefix)
    m_qb = match(r"qb(-?\d+)", prefix)
    zh = m_zh === nothing ? "?" : "$(parse(Int, m_zh.captures[1])/1000) m"
    qb_int = m_qb === nothing ? 0 : parse(Int, m_qb.captures[1])
    qb_str = qb_int == 0 ? "0" : @sprintf("%.0e", qb_int * 1e-9)
    return "zₕ = -$(zh)\nQ_b = $(qb_str)"
end

function stack(ts)
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
    @info "Loaded $nr stratified runs"

    data = [(stack(r.U), stack(r.B), stack(r.ww), stack(r.wb)) for r in runs]

    # Global colorranges per quantity
    Umax = maximum(maximum(d[1][3]) for d in data)
    # B is negative (b = -g(ρ-ρ₀)/ρ₀); use the most-negative across all runs
    Bmin = minimum(minimum(d[2][3]) for d in data)
    wpmax = maximum(maximum(d[3][3]) for d in data)
    wbmax = maximum(maximum(abs, d[4][3]) for d in data)

    fig = Figure(size=(450 * nr + 100, 1300), fontsize=14)
    rowlabels = ["U(z, t) [m/s]", "B(z, t) [m/s²]", "⟨w²⟩(z, t) [m²/s²]", "⟨w'b'⟩(z, t) [m²/s³]"]
    cmaps = (:thermal, :balance, :viridis, :balance)

    local hms = Vector{Any}(undef, 4)

    for (i, r) in enumerate(runs)
        for (j, ts) in enumerate((r.U, r.B, r.ww, r.wb))
            ts_t, zs, arr = data[i][j]
            ax = Axis(fig[j, i];
                      xlabel = j == 4 ? "t (s)" : "",
                      ylabel = i == 1 ? "z (m)" : "",
                      title = j == 1 ? shortlabel(r.prefix) : "")
            cmap = cmaps[j]
            crange = j == 1 ? (0, Umax) :
                     j == 2 ? (Bmin, -Bmin) :  # symmetric around 0
                     j == 3 ? (0, wpmax) : (-wbmax, wbmax)
            hm = heatmap!(ax, ts_t, zs, arr'; colormap=cmap, colorrange=crange)
            i == nr && (hms[j] = hm)
        end
    end

    for j in 1:4
        Colorbar(fig[j, nr + 1], hms[j]; label=rowlabels[j], width=18)
    end

    Label(fig[0, 1:nr+1], "Stratified LES matrix: U, B, ⟨w²⟩, ⟨w'b'⟩  (DNS-IC at t=19 s, run to t=60 s)";
          fontsize=15)

    save(out, fig; px_per_unit=2)
    @info "Saved $out"
end

if abspath(PROGRAM_FILE) == @__FILE__
    if endswith(ARGS[end], ".png")
        dirs = ARGS[1:end-1]
        out = ARGS[end]
    else
        dirs = ARGS
        out = "les_strat_matrix_compare.png"
    end
    compare_runs(dirs, out)
end
