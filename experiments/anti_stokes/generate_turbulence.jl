#####
##### Ambient turbulence initial conditions for the anti-Stokes campaign.
#####
##### Procedure (campaign document, section 8.2):
#####   1. random complex Fourier coefficients on a uniform triply periodic box;
#####   2. projection perpendicular to the wavevector (solenoidal);
#####   3. von Kármán energy spectrum with the energy-containing wavenumber k_e
#####      calibrated so that the streamwise integral scale matches the case;
#####      a Gaussian cutoff removes grid-scale noise;
#####   4. inverse transform, with each component phase-shifted onto its staggered face;
#####   5. linear interpolation onto the stretched Oceananigans grid;
#####   6. tapering of w near the top and bottom;
#####   7. component-wise rms scaling, then `set!` (pressure projection);
#####   8. a short no-wave preconditioning integration that forms the surface
#####      blocking layer;
#####   9. the velocity fields are saved to a reusable initial-condition file.
#####

using FFTW
using Random

isdefined(@__MODULE__, :RESOLUTION_LEVELS) || include("common.jl")

@inline von_karman_spectrum(k, k_e) = (k / k_e)^4 / (1 + (k / k_e)^2)^(17/6)

"""
    spectral_velocity_field(Nx, Ny, Nz, Lx, Ly, Lz; k_e, k_cut, rng)

Random solenoidal velocity field on a uniform triply periodic box with a von Kármán
energy spectrum peaking near `k_e`, cut off with `exp(-(k/k_cut)^2)`. Returns Float64
arrays `(u, v, w)` sampled on x-, y-, and z-faces respectively (each `Nx × Ny × Nz`).
The mean of every component is zero.
"""
function spectral_velocity_field(Nx, Ny, Nz, Lx, Ly, Lz; k_e, k_cut, rng)
    Δx, Δy, Δz = Lx / Nx, Ly / Ny, Lz / Nz

    kx = 2π / Lx .* collect(0:Nx÷2)
    ky = 2π / Ly .* collect(fftfreq(Ny, Ny))
    kz = 2π / Lz .* collect(fftfreq(Nz, Nz))

    KX = reshape(kx, :, 1, 1)
    KY = reshape(ky, 1, :, 1)
    KZ = reshape(kz, 1, 1, :)
    K² = @. KX^2 + KY^2 + KZ^2
    K²⁺ = copy(K²)
    K²⁺[1, 1, 1] = 1  # avoid division by zero at the mean
    K = sqrt.(K²)

    û = randn(rng, ComplexF64, size(K))
    v̂ = randn(rng, ComplexF64, size(K))
    ŵ = randn(rng, ComplexF64, size(K))

    # Project onto the plane perpendicular to the wavevector
    k_dot_u = @. (KX * û + KY * v̂ + KZ * ŵ) / K²⁺
    @. û -= KX * k_dot_u
    @. v̂ -= KY * k_dot_u
    @. ŵ -= KZ * k_dot_u

    # Shell-integrated spectrum E(k): |û(k)|² ∝ E(k) / (4π k²)
    amplitude = @. sqrt(von_karman_spectrum(K, k_e) * exp(-(K / k_cut)^2) / (4π * K²⁺))
    amplitude[1, 1, 1] = 0  # no mean momentum
    @. û *= amplitude
    @. v̂ *= amplitude
    @. ŵ *= amplitude

    # Sample each component on its own staggered face: f(x - Δ/2) ↔ f̂ exp(-i k Δ/2)
    @. û *= exp(-im * KX * Δx / 2)
    @. v̂ *= exp(-im * KY * Δy / 2)
    @. ŵ *= exp(-im * KZ * Δz / 2)

    u = irfft(û, Nx)
    v = irfft(v̂, Nx)
    w = irfft(ŵ, Nx)

    return u, v, w
end

"""
    streamwise_integral_scale(u, Δx)

Longitudinal integral scale of `u` along the (periodic) first dimension, averaged
over the remaining dimensions: the autocorrelation integrated to its first zero.
"""
function streamwise_integral_scale(u, Δx)
    Nx = size(u, 1)
    û = rfft(u, 1)
    S = dropdims(mean(abs2, û; dims=(2, 3)); dims=(2, 3))
    R = irfft(S, Nx)
    R ./= R[1]
    L = 0.5 * R[1] * Δx
    for r in 2:Nx÷2
        R[r] <= 0 && break
        L += R[r] * Δx
    end
    return L
end

"""
    calibrate_k_e(Nx, Ny, Nz, Lx, Ly, Lz; L_target, k_cut, seed, iterations=4)

Fixed-point iteration on the energy-containing wavenumber so that the generated
field (with the same random phases) has streamwise integral scale `L_target`.
"""
function calibrate_k_e(Nx, Ny, Nz, Lx, Ly, Lz; L_target, k_cut, seed, iterations=4)
    k_e = 0.75 / L_target  # exact for an untruncated von Kármán spectrum
    for _ in 1:iterations
        u, _, _ = spectral_velocity_field(Nx, Ny, Nz, Lx, Ly, Lz; k_e, k_cut, rng=Xoshiro(seed))
        L = streamwise_integral_scale(u, Lx / Nx)
        k_e *= L / L_target
    end
    return k_e
end

"""
    interpolate_columns(f, z_from, z_to)

Linear interpolation of `f` along its third dimension from the sorted coordinates
`z_from` to `z_to`, clamping outside the source range.
"""
function interpolate_columns(f, z_from, z_to)
    Nx, Ny, _ = size(f)
    g = zeros(eltype(f), Nx, Ny, length(z_to))
    for (k, z) in enumerate(z_to)
        j = clamp(searchsortedlast(z_from, z), 1, length(z_from) - 1)
        ω = clamp((z - z_from[j]) / (z_from[j+1] - z_from[j]), 0, 1)
        @views @. g[:, :, k] = (1 - ω) * f[:, :, j] + ω * f[:, :, j+1]
    end
    return g
end

"""
    parse_amplitude(s)

Parse `"1.3"` or `"1.3,1.0,1.1"` into a 3-tuple of per-component amplitude multipliers.
"""
function parse_amplitude(s::AbstractString)
    values = parse.(Float64, split(s, ','))
    length(values) == 1 && return (values[1], values[1], values[1])
    length(values) == 3 && return Tuple(values)
    error("amplitude must be one number or three comma-separated numbers, got \"$s\"")
end
parse_amplitude(a::Number) = (Float64(a), Float64(a), Float64(a))
parse_amplitude(a::Tuple) = Float64.(a)

"""
    generate_initial_condition(; kw...)

Generate, precondition, and save one ambient-turbulence initial condition. Returns the
path of the saved file. `amplitude` (a number or a 3-tuple for (u, v, w)) multiplies the
measured rms velocities to account for decay between t = 0 and the pre-packet observation
window; it is the calibration knob (campaign document, section 8.3). The component rms
values are enforced after the pressure projection by `projection_iterations` rounds of
rescaling and re-projection, because the projection onto the bounded, narrow tank
redistributes energy between components. `L_factor` multiplies the target integral scale
handed to the spectral generator (the scale shrinks during preconditioning and decay).
"""
function generate_initial_condition(; case_name = "1.D",
                                      level = "S0",
                                      seed = 1,
                                      arch = GPU(),
                                      FT = Float32,
                                      Lx = 12.0,
                                      Ly = 0.8,
                                      amplitude = 1.6,
                                      projection_iterations = 3,
                                      L_factor = 1.15,
                                      precondition_time = 2.0,
                                      Δt = 0.02,
                                      numerics = "weno",
                                      cutoff_cells = 6,
                                      taper_depth = 0.05,
                                      root = default_data_root(),
                                      overwrite = false,
                                      progress_interval = 20)

    case = anti_stokes_case(case_name, FT)
    path = initial_condition_path(root, case, level, seed; Lx, Ly)
    amplitude = parse_amplitude(amplitude)
    rms_target = (u = Float64(case.u_rms), v = Float64(case.v_rms), w = Float64(case.w_rms))
    rms_scaled = (u = amplitude[1] * rms_target.u, v = amplitude[2] * rms_target.v, w = amplitude[3] * rms_target.w)

    if isfile(path) && !overwrite
        @info "Initial condition $path already exists; skipping generation."
        return path
    end

    (; Nx, Ny, Nz) = level_size(level)
    Lz = Float64(case.h)
    num = numerics_settings(numerics, FT)
    grid = build_grid(arch, FT; Nx, Ny, Nz, Lx, Ly, Lz, halo=num.halo)

    #####
    ##### Spectral field on the auxiliary uniform box
    #####

    Δmax = max(Lx / Nx, Ly / Ny, Lz / Nz)
    k_cut = π / (cutoff_cells * Δmax)
    L_target = L_factor * Float64(case.L)

    @info "Calibrating energy-containing wavenumber for L = $L_target m (k_cut = $(round(k_cut, digits=2)) m⁻¹)..."
    k_e = calibrate_k_e(Nx, Ny, Nz, Lx, Ly, Lz; L_target, k_cut, seed)
    u, v, w = spectral_velocity_field(Nx, Ny, Nz, Lx, Ly, Lz; k_e, k_cut, rng=Xoshiro(seed))
    L_generated = streamwise_integral_scale(u, Lx / Nx)
    @info @sprintf("k_e = %.3f m⁻¹, generated L₁₁ = %.4f m (target %.4f m)", k_e, L_generated, L_target)

    # Component-wise rms scaling before tapering
    rms(a) = sqrt(mean(abs2, a))
    u .*= rms_scaled.u / rms(u)
    v .*= rms_scaled.v / rms(v)
    w .*= rms_scaled.w / rms(w)

    #####
    ##### Interpolate onto the stretched grid and taper w
    #####

    Δz = Lz / Nz
    zc_aux = [-Lz - Δz/2; [-Lz + (k - 0.5) * Δz for k in 1:Nz]; Δz/2]  # periodic wrap on both sides
    zf_aux = [-Lz + (k - 1) * Δz for k in 1:Nz+1]
    wrap_centers(a) = cat(a[:, :, end:end], a, a[:, :, 1:1]; dims=3)
    wrap_faces(a) = cat(a, a[:, :, 1:1]; dims=3)

    zc = Array(znodes(grid, Center()))
    zf = Array(znodes(grid, Face()))

    uᵢ = interpolate_columns(wrap_centers(u), zc_aux, zc)
    vᵢ = interpolate_columns(wrap_centers(v), zc_aux, zc)
    wᵢ = interpolate_columns(wrap_faces(w), zf_aux, zf)

    taper = @. tanh(-zf / taper_depth) * tanh((zf + Lz) / taper_depth)
    wᵢ .*= reshape(taper, 1, 1, :)
    wᵢ[:, :, 1] .= 0
    wᵢ[:, :, end] .= 0

    rms_spectral = (u = rms(uᵢ), v = rms(vᵢ), w = rms(wᵢ))

    #####
    ##### Project and precondition
    #####

    @info "Building preconditioning model on $(summary(grid))"
    model = build_model(grid; advection=num.advection, closure=num.closure)
    set!(model; u=FT.(uᵢ), v=FT.(vᵢ), w=FT.(wᵢ))
    rms_projected = component_rms(model)
    @info @sprintf("rms after projection: (%.4f, %.4f, %.4f) m s⁻¹", rms_projected...)

    # Enforce the component rms after projection: rescale and re-project
    for iteration in 1:projection_iterations
        r = component_rms(model)
        uₘ, vₘ, wₘ = model.velocities
        set!(model; u = Array(interior(uₘ)) .* FT(rms_scaled.u / r.u),
                    v = Array(interior(vₘ)) .* FT(rms_scaled.v / r.v),
                    w = Array(interior(wₘ)) .* FT(rms_scaled.w / r.w))
        r = component_rms(model)
        @info @sprintf("rms after rescaling + re-projection %d: (%.4f, %.4f, %.4f) m s⁻¹ (targets %.4f, %.4f, %.4f)",
                       iteration, r..., rms_scaled...)
    end
    rms_enforced = component_rms(model)

    simulation = Simulation(model; Δt, stop_time=precondition_time, verbose=false)
    simulation.callbacks[:progress] = Callback(make_progress(), IterationInterval(progress_interval))

    @info "Preconditioning for $precondition_time s..."
    wall = time_ns()
    run!(simulation)
    wall_time = 1e-9 * (time_ns() - wall)

    rms_final = component_rms(model)
    @info @sprintf("rms after preconditioning: (%.4f, %.4f, %.4f) m s⁻¹", rms_final...)

    uf, vf, wf = model.velocities
    u_out = Array(interior(uf))
    v_out = Array(interior(vf))
    w_out = Array(interior(wf))
    any(isnan, u_out) && error("NaN in preconditioned initial condition")

    metadata = (; case = case_name, level, seed, Nx, Ny, Nz, Lx, Ly, Lz, FT = string(FT),
                  amplitude, projection_iterations, L_factor, k_e, k_cut, cutoff_cells, taper_depth,
                  L_target, L_generated, precondition_time, Δt, numerics,
                  rms_target, rms_scaled, rms_spectral, rms_projected, rms_enforced, rms_final, wall_time,
                  oceananigans = oceananigans_version(), commit = git_commit(),
                  created = string(now()))

    mkpath(dirname(path))
    jldsave(path, false, IOStream; u=u_out, v=v_out, w=w_out, metadata)
    @info "Saved initial condition to $path"

    return path
end

function load_initial_condition(path)
    return jldopen(path) do file
        file["u"], file["v"], file["w"], file["metadata"]
    end
end

#####
##### Command-line entry point:
#####   julia --project=. experiments/anti_stokes/generate_turbulence.jl case=1.D level=S0 seed=1
#####

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_key_value_args(ARGS)
    generate_initial_condition(; case_name = getarg(args, "case", "1.D"),
                                 level = getarg(args, "level", "S0"),
                                 seed = getarg(args, "seed", 1),
                                 arch = architecture(getarg(args, "arch", "gpu")),
                                 FT = float_type(getarg(args, "FT", "Float32")),
                                 Lx = getarg(args, "Lx", 12.0),
                                 Ly = getarg(args, "Ly", 0.8),
                                 amplitude = parse_amplitude(getarg(args, "amplitude", "1.6")),
                                 projection_iterations = getarg(args, "projection_iterations", 3),
                                 L_factor = getarg(args, "L_factor", 1.15),
                                 precondition_time = getarg(args, "precondition_time", 2.0),
                                 Δt = getarg(args, "dt", 0.02),
                                 numerics = getarg(args, "numerics", "weno"),
                                 cutoff_cells = getarg(args, "cutoff_cells", 6),
                                 taper_depth = getarg(args, "taper_depth", 0.05),
                                 root = getarg(args, "root", default_data_root()),
                                 overwrite = getarg(args, "overwrite", false))
end
