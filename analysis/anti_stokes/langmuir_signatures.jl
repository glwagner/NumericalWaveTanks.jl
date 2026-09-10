#####
##### Signatures of Langmuir turbulence for a set of runs, from the y-averaged moments (y_averages.jld2,
##### further averaged in x) and the 3D snapshots (snapshots.jld2):
#####
#####   row 1  ⟨uᴸ⟩(z) (solid) and ⟨uᴱ⟩ = ⟨uᴸ⟩ − uˢ (dashed) at the requested times — is the Lagrangian mean homogenized?
#####   row 2  anisotropy ⟨w′²⟩/⟨u′²⟩(z): shear turbulence keeps it well below 1, Langmuir cells push it toward or above 1
#####   row 3  TKE production: Stokes production −⟨u′w′⟩∂zuˢ vs mean-shear production −⟨u′w′⟩∂z⟨uᴱ⟩ at the last time
#####   row 4  skewness of w from the snapshots: strong narrow downwelling → negative skewness in the Stokes layer
#####   row 5  (all runs) streamwise/spanwise integral scales of w at z = −3 cm from the snapshots, and the
#####          Stokes-layer-averaged ⟨w′²⟩ against time
#####
##### Usage: sbatch batch/anti_stokes_analysis.batch script=langmuir_signatures.jl runs=<d1>,<d2>,... "labels=a|b|..." times=5,10,20,40,60 [depth=0.03 name=langmuir_signatures]
#####

using CairoMakie
using FFTW
include("common.jl")

args = parse_key_value_args(ARGS)
dirs = String.(split(args["runs"], ','))
labels = String.(split(get(args, "labels", join(basename.(dirname.(dirs)), "|")), '|'))
times_req = parse.(Float64, split(getarg(args, "times", "5,10,20"), ','))
depth = getarg(args, "depth", 0.03)
name = getarg(args, "name", "langmuir_signatures")
nr = length(dirs)

xmean(A) = dropdims(mean(A; dims=1); dims=1)
function corr_length(A, Δ; dim)
    B = A .- mean(A); F = fft(B, dim); R = real.(ifft(abs2.(F), dim))
    R = dropdims(mean(R; dims = dim == 1 ? 2 : 1); dims = dim == 1 ? 2 : 1); R ./= R[1]
    n = findfirst(<=(0), R); n = isnothing(n) ? length(R) ÷ 2 : n
    return Δ * (0.5 + sum(R[2:n-1]; init=0.0))
end

results = []
for (d, lbl) in zip(dirs, labels)
    r = load_run(d)
    t, z, zf, k = times(r), znodes_centers(r), znodes_faces(r), k₀(r)
    UL = xmean(xzt(r, "U")); UE = xmean(eulerian_U(r))
    uˢz = xmean(stokes_U(r))
    m = central_moments(r)
    uu, ww, uw = xmean(m.uu), xmean(m.ww), xmean(m.uw)       # uu, uw at centers (Nz), ww at faces (Nz+1)
    ww_c = 0.5 .* (ww[1:end-1, :] .+ ww[2:end, :])
    aniso = ww_c ./ max.(uu, 1e-12)
    Δz = diff(zf)
    ∂z(P) = vcat((P[2:end, :] .- P[1:end-1, :]) ./ (z[2:end] .- z[1:end-1]), zeros(1, size(P, 2)))   # at upper faces, padded
    P_S = -uw .* ∂z(uˢz); P_M = -uw .* ∂z(UE)
    # Stokes-layer (k₀z > −2) averaged ⟨w′²⟩ against time
    layer = findall(zz -> k * zz > -2, z)
    wvar_layer = vec(sum(ww_c[layer, :] .* Δz[layer]; dims=1) ./ sum(Δz[layer]))
    # snapshots: skewness profiles and w elongation at depth
    f = jldopen(joinpath(d, "snapshots.jld2"))
    its = sort(parse.(Int, keys(f["timeseries/t"]))); ts = [f["timeseries/t/$i"] for i in its]
    grid = run_grid(r); kf = argmin(abs.(zf .+ depth))
    skew = Dict{Float64, Vector{Float64}}(); elong = Float64[]; elong_t = Float64[]
    for (i, tsnap) in zip(its, ts)
        w = Float64.(f["timeseries/w/$i"])
        wp = w .- mean(w; dims=(1, 2))
        s2 = dropdims(mean(wp .^ 2; dims=(1, 2)); dims=(1, 2)); s3 = dropdims(mean(wp .^ 3; dims=(1, 2)); dims=(1, 2))
        any(isapprox.(tsnap, times_req; atol=0.6)) && (skew[tsnap] = vec(s3 ./ max.(s2 .^ 1.5, 1e-18)))
        sl = w[:, :, kf]
        push!(elong, corr_length(sl, grid.Lx / grid.Nx; dim=1) / max(corr_length(sl, grid.Ly / grid.Ny; dim=2), 1e-9)); push!(elong_t, tsnap)
    end
    close(f)
    push!(results, (; lbl, t, z, zf, k, UL, UE, uˢz, aniso, P_S, P_M, wvar_layer, skew, elong, elong_t, member=r.meta["member"]))
    @info @sprintf("%s: t_end %.1f s; surface ⟨uᴸ⟩ at t=%s: %s mm/s", lbl, t[end], join(string.(times_req), "/"), join([@sprintf("%.1f", 1e3UL[end, nearest_index(t, tt)]) for tt in times_req], "/"))
end

set_theme!(Theme(fontsize=15))
fig = Figure(size = (520 * nr + 100, 2050))
Label(fig[0, 1:nr], "Langmuir-turbulence signatures: " * join(labels, " · "), fontsize=20)
colors = Makie.wong_colors()
for (j, R) in enumerate(results)
    kz = R.k .* R.z
    ax1 = Axis(fig[1, j]; title=R.lbl, xlabel="⟨uᴸ⟩ (solid), ⟨uᴱ⟩ (dashed) (mm/s)", ylabel = j == 1 ? "k₀ z" : "")
    for (i, tt) in enumerate(times_req)
        n = nearest_index(R.t, tt); tt <= R.t[end] + 1e-6 || continue
        lines!(ax1, 1e3 .* R.UL[:, n], kz; color=colors[mod1(i, 7)], linewidth=2.5, label="t = $(tt) s")
        lines!(ax1, 1e3 .* R.UE[:, n], kz; color=colors[mod1(i, 7)], linewidth=1.5, linestyle=:dash)
    end
    lines!(ax1, 1e3 .* R.uˢz[:, nearest_index(R.t, times_req[end])], kz; color=(:black, 0.4), linestyle=:dot, label="uˢ")
    vlines!(ax1, [0]; color=(:black, 0.3)); ylims!(ax1, -4, 0); j == nr && axislegend(ax1; position=:rb, labelsize=11)

    ax2 = Axis(fig[2, j]; xlabel="⟨w′²⟩ / ⟨u′²⟩", ylabel = j == 1 ? "k₀ z" : "")
    for (i, tt) in enumerate(times_req)
        tt <= R.t[end] + 1e-6 || continue
        lines!(ax2, R.aniso[:, nearest_index(R.t, tt)], kz; color=colors[mod1(i, 7)], linewidth=2.5, label="t = $(tt) s")
    end
    vlines!(ax2, [1]; color=(:black, 0.4), linestyle=:dot); xlims!(ax2, 0, 2.5); ylims!(ax2, -4, 0); j == nr && axislegend(ax2; position=:rb, labelsize=11)

    ax3 = Axis(fig[3, j]; xlabel="TKE production (m²/s³ ×10⁶) at t = $(times_req[end]) s", ylabel = j == 1 ? "k₀ z" : "")
    n = nearest_index(R.t, min(times_req[end], R.t[end]))
    lines!(ax3, 1e6 .* R.P_S[:, n], kz; color=colors[1], linewidth=2.5, label="Stokes: −⟨u′w′⟩∂zuˢ")
    lines!(ax3, 1e6 .* R.P_M[:, n], kz; color=colors[2], linewidth=2.5, label="mean shear: −⟨u′w′⟩∂z⟨uᴱ⟩")
    vlines!(ax3, [0]; color=(:black, 0.3)); ylims!(ax3, -4, 0); j == nr && axislegend(ax3; position=:rb, labelsize=11)

    ax4 = Axis(fig[4, j]; xlabel="skewness of w (snapshots)", ylabel = j == 1 ? "k₀ z" : "")
    for (i, tt) in enumerate(sort(collect(keys(R.skew))))
        lines!(ax4, R.skew[tt], R.k .* R.zf; color=colors[mod1(i, 7)], linewidth=2.5, label=@sprintf("t = %.0f s", tt))
    end
    vlines!(ax4, [0]; color=(:black, 0.3)); xlims!(ax4, -2, 2); ylims!(ax4, -4, 0); j == nr && !isempty(R.skew) && axislegend(ax4; position=:rb, labelsize=11)
end
ax5 = Axis(fig[5, 1:max(1, nr ÷ 2)]; xlabel="t (s)", ylabel="L_x / L_y of w at z = −$(round(Int, 100depth)) cm", title="streamwise elongation of w (snapshots)")
ax6 = Axis(fig[5, max(1, nr ÷ 2)+1:nr]; xlabel="t (s)", ylabel="⟨w′²⟩ averaged over k₀z > −2 (mm²/s²)", title="Stokes-layer vertical-velocity variance")
for (j, R) in enumerate(results)
    scatterlines!(ax5, R.elong_t, R.elong; color=colors[mod1(j, 7)], linewidth=2, label=R.lbl)
    lines!(ax6, R.t, 1e6 .* R.wvar_layer; color=colors[mod1(j, 7)], linewidth=2.5, label=R.lbl)
end
hlines!(ax5, [1]; color=(:black, 0.3)); axislegend(ax5; position=:lt, labelsize=11); axislegend(ax6; position=:rt, labelsize=11)
output = joinpath(figure_directory(), "$(name).png")
save(output, fig)
@info "Saved $output"
