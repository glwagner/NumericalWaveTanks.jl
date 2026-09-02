#####
##### Makie-free utilities for analysing anti-Stokes campaign output: loading runs,
##### Eulerian subtraction of the prescribed Stokes drift, central moments from the
##### y-averaged second moments, window averages, and scalar statistics.
#####

using Oceananigans
using JLD2
using Statistics
using Printf

# The experiment code (configuration, packet, turbulence generator) may already be loaded,
# e.g. by stage_S0.jl or the smoke test; generate_turbulence.jl includes common.jl itself.
isdefined(@__MODULE__, :streamwise_integral_scale) ||
    include(joinpath(@__DIR__, "..", "..", "experiments", "anti_stokes", "generate_turbulence.jl"))

struct Run
    dir  :: String
    meta :: Dict{String, Any}
    fts  :: Dict{String, Any}
end

Base.show(io::IO, run::Run) = print(io, "Run(", run.meta["member"], ", ", run.dir, ")")

"""
    load_run(dir; fields=("U", "V", "W", "UU", "VV", "WW", "UW"))

Load metadata and the requested y-averaged `FieldTimeSeries` from a run directory.
"""
function load_run(dir; fields=("U", "V", "W", "UU", "VV", "WW", "UW"))
    isdir(dir) || error("Run directory $dir does not exist")
    meta = load(joinpath(dir, "metadata.jld2"))
    path = joinpath(dir, "y_averages.jld2")
    fts = Dict{String, Any}(name => FieldTimeSeries(path, name) for name in fields)
    return Run(dir, meta, fts)
end

run_packet(run)    = run.meta["packet"]
run_case(run)      = run.meta["case"]
times(run)         = collect(Float64, run.fts[first(keys(run.fts))].times)
run_grid(run)      = run.fts[first(keys(run.fts))].grid
# With end walls x-face fields carry Nx + 1 points; the last (wall) face is dropped so that
# face arrays line up with the Nx cell centres as in the periodic layout.
function xnodes_faces(run)
    xf = collect(Float64, Array(xnodes(run_grid(run), Face())))
    return length(xf) == run.meta["Nx"] + 1 ? xf[1:end-1] : xf
end
xnodes_centers(run) = collect(Float64, Array(xnodes(run_grid(run), Center())))
znodes_centers(run) = collect(Float64, Array(znodes(run_grid(run), Center())))
znodes_faces(run)  = collect(Float64, Array(znodes(run_grid(run), Face())))
Δz_centers(run)    = diff(znodes_faces(run))
fov_index(run)     = run.meta["i_FOV"]
τ₀(run)            = Float64(run_case(run).τ₀)
t_peak(run)        = Float64(run.meta["t_peak"])
k₀(run)            = Float64(run_case(run).k)
windows(run)       = analysis_windows(t_peak(run), τ₀(run))
before_window(run) = windows(run).before
after_window(run)  = windows(run).after
passage_window(run) = windows(run).passage
is_bounded_x(run)  = get(run.meta, "x_topology", "periodic") == "bounded"

"""
    xzt(run, name)

The y-averaged field `name` as a `(Nx, Nz, Nt)` array (Nz + 1 for z-face fields).
"""
function xzt(run, name)
    A = Array{Float64}(Array(interior(run.fts[name]))[:, 1, :, :])
    return size(A, 1) == run.meta["Nx"] + 1 ? A[1:end-1, :, :] : A
end

"""
    eulerian_U(run)

⟨u^E⟩_y = ⟨u^L⟩_y − uˢ(x, z, t), using the packet parameters stored in the metadata.
For members without a packet this is just ⟨u^L⟩_y.
"""
function eulerian_U(run)
    U = xzt(run, "U")
    run.meta["has_packet"] || return U
    p = run_packet(run)
    x, z, t = xnodes_faces(run), znodes_centers(run), times(run)
    for n in eachindex(t), k in eachindex(z), i in eachindex(x)
        U[i, k, n] -= uˢ(x[i], 0, z[k], t[n], p)
    end
    return U
end

function eulerian_W(run)
    W = xzt(run, "W")
    run.meta["has_packet"] || return W
    p = run_packet(run)
    x, z, t = xnodes_centers(run), znodes_faces(run), times(run)
    for n in eachindex(t), k in eachindex(z), i in eachindex(x)
        W[i, k, n] -= wˢ(x[i], 0, z[k], t[n], p)
    end
    return W
end

"""
    stokes_U(run)

The prescribed uˢ on the same (x-face, z-center, t) points as `xzt(run, "U")`.
"""
function stokes_U(run)
    x, z, t = xnodes_faces(run), znodes_centers(run), times(run)
    run.meta["has_packet"] || return zeros(length(x), length(z), length(t))
    p = run_packet(run)
    return [uˢ(xi, 0, zk, tn, p) for xi in x, zk in z, tn in t]
end

"""
    central_moments(run)

Eulerian turbulent variances and covariance from the y-averaged moments:
u'² = ⟨u²⟩ − ⟨u⟩² (x-faces), v'² (cell centers), w'² (z-faces), and
u'w' = ⟨uw⟩ − ⟨u⟩⟨w⟩ at cell centers. Subtracting the y-uniform Stokes drift
does not change these.
"""
function central_moments(run)
    U, V, W = xzt(run, "U"), xzt(run, "V"), xzt(run, "W")
    UU, VV, WW, UW = xzt(run, "UU"), xzt(run, "VV"), xzt(run, "WW"), xzt(run, "UW")
    uu = UU .- U.^2
    vv = VV .- V.^2
    ww = WW .- W.^2
    Uc = 0.5 .* (U .+ circshift(U, (-1, 0, 0)))          # x-face → cell center (periodic)
    Wc = 0.5 .* (W[:, 1:end-1, :] .+ W[:, 2:end, :])     # z-face → cell center
    uw = UW .- Uc .* Wc
    return (; uu, vv, ww, uw)
end

"""
    window_mean(A, t, t₀, t₁)

Time average of the `(Nx, Nz, Nt)` array `A` over output times in `[t₀, t₁]`.
"""
function window_mean(A, t, t₀, t₁)
    idx = findall(τ -> t₀ - 1e-6 <= τ <= t₁ + 1e-6, t)
    isempty(idx) && error("No output times in [$t₀, $t₁]")
    return dropdims(mean(A[:, :, idx]; dims=3); dims=3)
end

nearest_index(t, t₀) = argmin(abs.(t .- t₀))

"""
    load_statistics(dir)

Scalar time series written by the `statistics` output writer as a `Dict`
with key `"t"` plus one entry per statistic.
"""
function load_statistics(dir)
    path = joinpath(dir, "statistics.jld2")
    return jldopen(path) do file
        iters = sort([parse(Int, k) for k in keys(file["timeseries/t"]) if all(isdigit, k)])
        names = [k for k in keys(file["timeseries"]) if k != "t"]
        stats = Dict{String, Vector{Float64}}("t" => [file["timeseries/t/$i"] for i in iters])
        for name in names
            stats[name] = [Float64(file["timeseries/$name/$i"]) for i in iters]
        end
        return stats
    end
end

"""
    load_snapshot(dir, name, t)

Interior array of the 3D snapshot of `name` nearest time `t`, and the actual time.
"""
function load_snapshot(dir, name, t)
    fts = FieldTimeSeries(joinpath(dir, "snapshots.jld2"), name)
    n = nearest_index(collect(fts.times), t)
    return Array{Float64}(Array(interior(fts[n]))), Float64(fts.times[n])
end

"""
    integral_scale_profile(u, Δx)

Streamwise integral scale at every vertical level of a 3D array `u`, averaging the
autocorrelation over the spanwise direction.
"""
function integral_scale_profile(u, Δx)
    Nx, Ny, Nz = size(u)
    L = zeros(Nz)
    for k in 1:Nz
        L[k] = streamwise_integral_scale(u[:, :, k:k], Δx)
    end
    return L
end

"""
    depth_integral(profile, Δz)

∫ profile dz with the cell thicknesses `Δz`, and the positive and negative lobes.
"""
function depth_integral(profile, Δz)
    total = sum(profile .* Δz)
    positive = sum(max.(profile, 0) .* Δz)
    negative = sum(min.(profile, 0) .* Δz)
    return (; total, positive, negative)
end

"""
    paired_residual(packet_dir, control_dir, null_dir=nothing, quiescent_dir=nothing)

ΔU_turb = (U_packet+turb − U_turb) − (U_packet+null − U_null) as an `(Nx, Nz, Nt)` array,
together with the packet and control `Run`s.
"""
function paired_residual(packet_dir, control_dir, null_dir=nothing, quiescent_dir=nothing)
    pk = load_run(packet_dir)
    ct = load_run(control_dir)
    t = times(pk)
    t ≈ times(ct) || error("Packet and control output times differ")
    ΔU = xzt(pk, "U") .- xzt(ct, "U")

    if !isnothing(null_dir)
        nl = load_run(null_dir; fields=("U",))
        times(nl) ≈ t || error("Null output times differ from the pair")
        U_null = xzt(nl, "U")
        if !isnothing(quiescent_dir)
            qc = load_run(quiescent_dir; fields=("U",))
            U_null .-= xzt(qc, "U")
        end
        ΔU .-= U_null
    end

    return ΔU, pk, ct
end

"""
    wake_age_composite(ΔU, x, t, p; age_edges)

Composite of the `(Nx, Nz, Nt)` array `ΔU` over all `(x, t)` with the same packet age
`a = (x_c(t) − x) / cᵍ` (positive behind the packet, negative ahead of it), binned by
`age_edges`. In the current-following frame the wake is a spatial record of the temporal
response, so each age bin averages over ~c_g t_stop / L ≈ 60 integral scales rather than the
one or two eddies that cross the fixed observation plane in a window. Returns
`(ages, composite (Nz, Na), counts)`.
"""
function wake_age_composite(ΔU, x, t, p; age_edges)
    Nx, Nz, Nt = size(ΔU)
    Na = length(age_edges) - 1
    C = zeros(Nz, Na)
    N = zeros(Int, Na)
    for n in 1:Nt
        xc = packet_center(t[n], p)
        for i in 1:Nx
            a = (xc - x[i]) / p.cᵍ
            b = searchsortedlast(age_edges, a)
            1 <= b <= Na || continue
            a < age_edges[b+1] || continue
            @views C[:, b] .+= ΔU[i, :, n]
            N[b] += 1
        end
    end
    C ./= max.(N', 1)
    C[:, N .== 0] .= NaN          # bins with no samples
    ages = 0.5 .* (age_edges[1:end-1] .+ age_edges[2:end])
    return ages, C, N
end

default_age_edges(τ) = collect(range(-4τ, 16τ; step=τ / 4))

"""
    composite_profile(ages, C, τ, a₀, a₁)

Mean of the composite over age bins with centres in `[a₀, a₁]` (multiples of τ₀).
"""
function composite_profile(ages, C, τ, a₀, a₁)
    idx = findall(a -> a₀ * τ - 1e-9 <= a <= a₁ * τ + 1e-9, ages)
    idx = filter(i -> !any(isnan, C[:, i]), idx)
    isempty(idx) && error("No populated age bins in [$a₀, $a₁] τ₀")
    return vec(mean(C[:, idx]; dims=2))
end

symmetric_range(A; q=0.99) = (lim = quantile(abs.(vec(A)), q); lim == 0 ? (-1e-9, 1e-9) : (-lim, lim))

figure_directory() = mkpath(normpath(joinpath(@__DIR__, "..", "..", "figures", "anti_stokes")))
