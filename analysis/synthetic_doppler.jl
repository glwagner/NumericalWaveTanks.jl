using CairoMakie
using Oceananigans
using Statistics
using Printf

# Synthetic Doppler-shift observations from LES U(z, t).
#
#     c̃(k) = ∫_{-H}^0 U(z) Q(z, k) dz,    Q(z, k) = 2k exp(2kz)
#
# (Stewart & Joy 1974; Smeltzer et al. 2019). Each surface-wave wavenumber
# samples the current at an effective depth z_eff = -1/(2k). This script
# saves the c̃(k, t) timeseries computed from the LES — these are the
# "ground truth" synthetic observations a Doppler inversion (PEDM, EKI)
# would try to invert back to U(z).
#
# Usage:
#   julia --project synthetic_doppler.jl <run_dir> [out.png]

function load_U(dir)
    prefix = basename(rstrip(dir, '/'))
    f = joinpath(dir, prefix * "_profiles.jld2")
    isfile(f) || error("missing $f")
    return FieldTimeSeries(f, "U"), prefix
end

function compute_doppler(dir; ks = exp.(range(log(1.0), log(1000.0), length=80)))
    U_ts, prefix = load_U(dir)
    times = U_ts.times
    z_centers = collect(znodes(U_ts))
    Nz = length(z_centers)
    Nt = length(times)

    # Build cell widths for the integration. Use cell-face spacings.
    # For a stretched grid we need Δz at each center.
    Δz = zeros(Float64, Nz)
    for k in 2:Nz-1
        Δz[k] = 0.5 * (z_centers[k+1] - z_centers[k-1])
    end
    Δz[1] = z_centers[2] - z_centers[1]
    Δz[Nz] = z_centers[Nz] - z_centers[Nz-1]

    Nk = length(ks)
    c_tilde = zeros(Float64, Nk, Nt)
    z_eff   = -1.0 ./ (2 .* ks)

    for n in 1:Nt
        Uvec = vec(interior(U_ts[n]))
        for (j, k) in enumerate(ks)
            # Q(z, k) = 2k exp(2kz). Trapezoidal-weighted sum:
            s = 0.0
            for i in 1:Nz
                s += Uvec[i] * (2k * exp(2 * k * z_centers[i])) * Δz[i]
            end
            c_tilde[j, n] = s
        end
    end

    return (; times, ks, z_eff, c_tilde, prefix)
end

function plot_doppler(dirs, out)
    runs = [compute_doppler(d) for d in dirs]
    nr = length(runs)

    fig = Figure(size=(450 * nr + 100, 800), fontsize=14)

    # Panel 1: Hovmöller c̃(k, t) — log-k axis
    cmax = maximum(maximum(abs, r.c_tilde) for r in runs)
    for (i, r) in enumerate(runs)
        ax = Axis(fig[1, i];
                  xlabel = "t (s)",
                  ylabel = i == 1 ? "wavenumber k (1/m)" : "",
                  yscale = log10,
                  title = r.prefix)
        hm = heatmap!(ax, r.times, r.ks, r.c_tilde'; colormap=:thermal, colorrange=(0, cmax))
        i == nr && Colorbar(fig[1, nr+1], hm; label="c̃ (m/s)", width=18)
    end

    # Panel 2: c̃(k) at the final time, by run, plus z_eff axis
    ax = Axis(fig[2, 1:nr];
              xlabel = "wavenumber k (1/m) — top axis: z_eff = -1/(2k) (m)",
              ylabel = "c̃ (m/s)",
              xscale = log10)
    for r in runs
        lines!(ax, r.ks, r.c_tilde[:, end]; label = r.prefix, linewidth=2)
    end
    axislegend(ax; position=:rt, fontsize=9)

    Label(fig[0, 1:nr+1], "Synthetic Doppler observations  c̃(k, t) = ∫ U(z) · 2k e^(2kz) dz";
          fontsize=14)

    save(out, fig; px_per_unit=2)
    @info "Saved $out"
end

if abspath(PROGRAM_FILE) == @__FILE__
    if endswith(ARGS[end], ".png")
        dirs = ARGS[1:end-1]
        out = ARGS[end]
    else
        dirs = ARGS
        out = "synthetic_doppler.png"
    end
    plot_doppler(dirs, out)
end
