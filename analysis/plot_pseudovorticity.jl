using CairoMakie
using Oceananigans
using Printf

# Pseudovorticity number Ω(z, t) = ∂z u_L / ∂z u_S = 1 + ∂z U / ∂z u_S
#
# Interpretation (per claude_code_plan.md):
#   Ω → 1: Eulerian shear negligible (purely-wave-driven Stokes shear regime)
#   Ω ≫ 1: Eulerian shear adds to Stokes shear (laminar wind-drift, pre-instability)
#   Ω → 0 or negative: Eulerian shear opposes Stokes shear (well-mixed jet regime)
#
# The Stokes shear is analytic from the wave parameters:
#   ∂z u_S(z) = 2 ω ε² exp(2 k z)
# (using a = ε / k and the deep-water capillary-gravity ω = √(g k + γ k³)).
#
# Usage:
#   julia --project plot_pseudovorticity.jl <run_dir1> [<run_dir2> ...] [out.png]

function load_U(dir)
    prefix = basename(rstrip(dir, '/'))
    f = joinpath(dir, prefix * "_profiles.jld2")
    isfile(f) || error("missing $f")
    U = FieldTimeSeries(f, "U")
    return (; U, prefix)
end

function eps_from_prefix(prefix)
    m = match(r"ep(\d+)", prefix)
    m === nothing && return 0.11
    return parse(Int, m.captures[1]) / 1000
end

function stokes_shear(z, ϵ; k=2π/0.03, g=9.81, γ=7.2e-5)
    ω = sqrt(g * k + γ * k^3)
    return 2 * ω * ϵ^2 * exp(2 * k * z)
end

# ∂z U at face nodes via finite differences
function dz_U(U_arr, z_centers)
    Nz, Nt = size(U_arr)
    dz_U_face = zeros(Float32, Nz - 1, Nt)
    for n in 1:Nt, k in 1:Nz-1
        dz_U_face[k, n] = (U_arr[k+1, n] - U_arr[k, n]) / (z_centers[k+1] - z_centers[k])
    end
    z_faces = 0.5f0 * (z_centers[1:end-1] .+ z_centers[2:end])
    return dz_U_face, z_faces
end

function compute_omega(dir)
    r = load_U(dir)
    ϵ = eps_from_prefix(r.prefix)
    times = r.U.times
    z = znodes(r.U)
    Nz = length(z)
    Nt = length(times)
    U_arr = zeros(Float32, Nz, Nt)
    for n in 1:Nt
        U_arr[:, n] = vec(interior(r.U[n]))
    end

    dzU, zf = dz_U(U_arr, z)
    # Stokes shear at the same face nodes
    dz_uS = Float32[stokes_shear(zfk, ϵ) for zfk in zf]
    # Ω(z, t) = 1 + dzU / dz_uS (broadcast across time)
    Ω = 1f0 .+ dzU ./ dz_uS

    return (; times, zf, Ω, U=U_arr, z=z, ϵ, prefix=r.prefix)
end

function plot_omega(dirs, out)
    runs = [compute_omega(d) for d in dirs]
    nr = length(runs)

    # log10(Ω) — at lab scale wind-driven Eulerian shear dominates, so Ω≫1
    # everywhere; log scale exposes the structure
    log_Ω = [(log10.(max.(r.Ω, 1f-3))) for r in runs]
    log_max = maximum(maximum(x) for x in log_Ω)
    log_min = max(minimum(minimum(x) for x in log_Ω), -2)
    crange = (log_min, log_max)

    fig = Figure(size=(450 * nr + 100, 450), fontsize=14)
    # Stokes drift e-folding depth is 1/(2k) ≈ 2.4 mm; below ~5 cm the Stokes
    # shear is effectively zero, so Ω is numerically meaningless there. Limit
    # the plot to the active near-surface region.
    z_min_plot = -0.05
    local hm_ref
    for (i, r) in enumerate(runs)
        ax = Axis(fig[1, i];
                  xlabel = "t (s)",
                  ylabel = i == 1 ? "z (m)" : "",
                  title = @sprintf("ε = %.3g", r.ϵ),
                  limits = (nothing, nothing, z_min_plot, 0))
        hm = heatmap!(ax, r.times, r.zf, log_Ω[i]'; colormap=:viridis, colorrange=crange)
        i == 1 && (hm_ref = hm)
        contour!(ax, r.times, r.zf, r.Ω'; levels=[1f0], color=:black, linewidth=1.5)
    end
    Colorbar(fig[1, nr + 1], hm_ref; label="log₁₀ Ω", width=18)
    Label(fig[0, 1:nr+1], "Pseudovorticity number  Ω = ∂z u_L / ∂z u_S  (log scale)\n" *
          "Ω≫1 → Eulerian shear dominates;  Ω→1 → Stokes-dominated;  black contour: Ω=1";
          fontsize=13)

    save(out, fig; px_per_unit=2)
    @info "Saved $out"
end

if abspath(PROGRAM_FILE) == @__FILE__
    if endswith(ARGS[end], ".png")
        dirs = ARGS[1:end-1]
        out = ARGS[end]
    else
        dirs = ARGS
        out = "pseudovorticity.png"
    end
    plot_omega(dirs, out)
end
