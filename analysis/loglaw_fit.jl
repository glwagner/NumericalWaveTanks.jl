using CairoMakie
using Oceananigans
using Statistics
using LinearAlgebra: dot
using Printf

# Quasi-steady log-law fit to LES mean velocity profile.
#
# Model:    U(z) = (u*_eff / κ) log(-z / z₀_eff)
# Procedure:
#   1) Time-average U(z, t) over a quasi-steady window (default t > 40 s).
#   2) Restrict to a "log-layer" window in -z (default 5 mm < -z < 30 mm).
#   3) Linear least-squares fit  U  =  (u*_eff/κ) ln(-z) - (u*_eff/κ) ln(z₀_eff).
#   4) Compute nondimensional shear φ_m(z) = (κ z / u*_eff) dU/dz; deviation
#      from 1 is the wave-modified contribution.
#
# The fit gives u*_eff and z₀_eff regardless of wave forcing — but their
# values are what we expect to depend on the turbulent Langmuir number Laₜ
# (per Teixeira 2018 etc.).
#
# Usage:
#   julia --project plot_loglaw_fit.jl <run_dir1> [<run_dir2> ...] [out.png]

const κ_vk = 0.4   # von Kármán constant

function load_U(dir)
    prefix = basename(rstrip(dir, '/'))
    f = joinpath(dir, prefix * "_profiles.jld2")
    isfile(f) || error("missing $f")
    return FieldTimeSeries(f, "U"), prefix
end

function eps_from_prefix(prefix)
    m = match(r"ep(\d+)", prefix)
    m === nothing && return NaN
    return parse(Int, m.captures[1]) / 1000
end

function laminar_ustar(α; ν=1.05e-6, t=60.0)
    # τ(t) = α√t. Friction velocity u* = sqrt(τ).
    return sqrt(α * sqrt(t))
end

function quasi_steady_profile(dir; t_min=40.0)
    U_ts, prefix = load_U(dir)
    times = U_ts.times
    z = znodes(U_ts)
    Nz = length(z)
    mask = times .>= t_min
    Nm = sum(mask)
    Nm > 0 || error("no snapshots after t = $t_min s in $dir")
    Ū = zeros(Float64, Nz)
    for n in eachindex(times)
        mask[n] || continue
        Ū .+= vec(interior(U_ts[n]))
    end
    Ū ./= Nm
    return (; Ū, z = collect(z), prefix, ϵ = eps_from_prefix(prefix))
end

function fit_loglaw(z, U; z_min=-0.030, z_max=-0.005)
    # Restrict to log-layer window
    idx = findall(zk -> z_min <= zk <= z_max, z)
    isempty(idx) && error("empty fit window")
    Z = -z[idx]                 # positive depths
    Y = U[idx]
    X = log.(Z)
    # Linear LS: Y = a + b X  with  b = u*/κ,  a = -(u*/κ) ln(z₀)
    n = length(X)
    Xb = sum(X) / n
    Yb = sum(Y) / n
    SSxx = sum((X .- Xb).^2)
    SSxy = sum((X .- Xb) .* (Y .- Yb))
    b = SSxy / SSxx
    a = Yb - b * Xb
    # Wind-driven aquatic BL:  U(z) = U(0) - (u*/κ) log(-z/z₀)
    # so dU / d log(-z) = -u*/κ → u* = -b κ
    u_star = -b * κ_vk
    # U(z) = a + b X = (a - b log z₀) — but we don't have a separate U(0);
    # instead extrapolate the fit to z = -z₀ where U = U(0). With b<0:
    # at U_surface (extrapolated) the log argument is (-z/z₀) = 1 → z₀ = exp((-a+U(0))/(-b))
    # but since U(0) isn't well-defined here, report z₀ from the fit invariant:
    # the depth where the log fit predicts U = 0:
    z0 = exp(-a / b)
    Yhat = a .+ b .* X
    R2 = 1 - sum((Y .- Yhat).^2) / sum((Y .- Yb).^2)
    return (; u_star, z0, R2, idx, Z, Y, fit_z = Z, fit_U = Yhat)
end

function plot_fits(dirs, out; t_min = 40.0, z_min = -0.040, z_max = -0.002)
    runs = [quasi_steady_profile(d; t_min) for d in dirs]
    fits = [fit_loglaw(r.z, r.Ū; z_min, z_max) for r in runs]

    nr = length(runs)
    fig = Figure(size=(450 * nr + 100, 800), fontsize=14)

    Label(fig[0, 1:nr], @sprintf("Quasi-steady U(z) (t > %.0f s) and log-law fit  ζ = -z, U = (u*/κ) log(ζ/z₀)", t_min);
          fontsize=14)

    for (i, (r, f)) in enumerate(zip(runs, fits))
        u_star_lam = laminar_ustar(1.2e-5)

        ax = Axis(fig[1, i];
                  xlabel = "U (m/s)",
                  ylabel = i == 1 ? "z (m)" : "",
                  title = @sprintf("ε = %.3g\nu*_eff = %.4f m/s\nz₀ = %.2e m\nR² = %.3f",
                                   r.ϵ, f.u_star, f.z0, f.R2))
        # Profile
        lines!(ax, r.Ū, r.z; color=:black, label="LES (t-avg)")
        # Fit overlay
        z_fit = r.z[f.idx]
        U_fit = f.fit_U  # log values evaluated at the same X
        lines!(ax, U_fit, z_fit; color=:red, linewidth=2, linestyle=:dash, label="log-law fit")
        # Mark the fit window
        hlines!(ax, [z_min, z_max]; color=:gray, linestyle=:dot)
        ylims!(ax, -0.10, 0.0)

        if i == 1
            axislegend(ax; position=:rb, fontsize=10)
        end

        # Phi_m on the row below
        ax2 = Axis(fig[2, i];
                   xlabel = "φ_m = (κ z / u*) dU/dz",
                   ylabel = i == 1 ? "z (m)" : "",
                   title = "Nondimensional shear (φ_m = 1 → log law)")
        # Compute dU/dz numerically
        dz_U = diff(r.Ū) ./ diff(r.z)
        z_face = 0.5 .* (r.z[1:end-1] .+ r.z[2:end])
        # Restrict to z < 0
        valid = findall(zk -> -0.05 < zk < -0.001, z_face)
        # Wind-driven sign: dU/dz < 0 (U max at surface). Standard φ_m
        # convention takes |dU/dz|, so flip sign here.
        ϕ_m = κ_vk .* (-z_face[valid]) ./ f.u_star .* (-dz_U[valid])
        lines!(ax2, ϕ_m, z_face[valid]; color=:black)
        vlines!(ax2, [1.0]; color=:gray, linestyle=:dash)
        ylims!(ax2, -0.05, 0.0)
        xlims!(ax2, -0.5, 4.0)
    end

    save(out, fig; px_per_unit=2)
    @info "Saved $out"

    @info "u*_eff vs ε:"
    for (r, f) in zip(runs, fits)
        @info @sprintf("  ε = %.3g  →  u*_eff = %.4f m/s,  z₀_eff = %.2e m,  R² = %.3f",
                       r.ϵ, f.u_star, f.z0, f.R2)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if endswith(ARGS[end], ".png")
        dirs = ARGS[1:end-1]
        out = ARGS[end]
    else
        dirs = ARGS
        out = "loglaw_fit.png"
    end
    plot_fits(dirs, out)
end
