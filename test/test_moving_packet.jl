using Test
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Statistics
using Random

include(joinpath(@__DIR__, "..", "experiments", "anti_stokes", "forced_turbulence.jl"))
include(joinpath(@__DIR__, "..", "experiments", "anti_stokes", "moving_packet_experiment.jl"))

@testset "Case 1.D derived quantities" begin
    case = case_1D(Float64)
    @test case.c   ≈ 1.027  atol=1e-3
    @test case.cᵍ  ≈ 0.514  atol=1e-3
    @test case.Uˢ₀ ≈ 0.0497 atol=1e-4
    @test case.σ₀  ≈ 1.438  atol=1e-3
    @test case.δˢ  ≈ 0.0538 atol=1e-4
    @test case.Tᵢ  ≈ 4.963  atol=1e-3
    @test anti_stokes_case("1.D").name == "1.D"
    @test anti_stokes_case("1D", Float32).k isa Float32
    @test_throws ErrorException anti_stokes_case("9.Z")

    # Other wave-group cases (Ellingsen et al. 2026, Tables 2 and 3)
    for name in WAVE_GROUP_CASES
        c = anti_stokes_case(name, Float64)
        @test c.name == name
        @test c.h == 0.40
        @test c.c ≈ sqrt(9.81 / c.k)
        @test c.σ₀ ≈ c.cᵍ * c.τ₀
        @test c.Uˢ₀ ≈ c.steepness^2 * c.c
        # tabulated surface Stokes drift (Table 3, cm/s) within rounding
        @test 0 < c.Uˢ₀ < 0.06
        # The packet centre starts 4σ₀ upstream of the observation plane; for the 1.C cases
        # (σ₀ = 1.52 m) that is 9 cm past the periodic boundary, which the periodic image
        # sum handles. At t_stop the wrapped leading tail is still ≥ 3.9σ₀ from the plane.
        p = packet_parameters(c, 12.0, 6.0)
        @test p.x₀ > -0.1 * c.σ₀
        @test 6.0 + 12.0 - packet_center(packet_stop_time(p), p) > 3.8 * c.σ₀
    end
    @test anti_stokes_case("1.A", Float64).Uˢ₀ ≈ 0.041 atol=0.002
    @test anti_stokes_case("1.C.1", Float64).Uˢ₀ ≈ 0.023 atol=0.002
    @test anti_stokes_case("1.C.2", Float64).Uˢ₀ ≈ 0.051 atol=0.002
    @test anti_stokes_case("1.C.1", Float64).u_rms == anti_stokes_case("1.C.2", Float64).u_rms
end

@testset "Packet trajectory and observation frame" begin
    case = case_1D(Float64)
    p = packet_parameters(case, 12.0, 6.0)
    τ₀ = case.τ₀

    @test p.x₀ ≈ 0.249 atol=1e-3
    @test packet_peak_time(p) ≈ 4τ₀
    @test packet_stop_time(p) ≈ 8τ₀
    @test packet_stop_time(p) ≈ 22.4
    @test packet_center(packet_stop_time(p), p) ≈ 11.751 atol=1e-3

    # Stokes envelope at the FOV is ≤ e⁻⁹ of its peak in the before and after windows
    for t in vcat(range(0, τ₀, length=20), range(7τ₀, 8τ₀, length=20))
        @test uˢ(6.0, 0.0, 0.0, t, p) ≤ 1.3e-4 * case.Uˢ₀
    end
    @test uˢ(6.0, 0.0, 0.0, packet_peak_time(p), p) ≈ case.Uˢ₀

    # Envelope seen at the FOV: Uˢ₀ exp[-(t - t_peak)² / τ₀²]
    for t in range(0, 22.4, length=15)
        @test uˢ(6.0, 0.0, 0.0, t, p) ≈ case.Uˢ₀ * exp(-(t - 11.2)^2 / τ₀^2) atol=1e-7 * case.Uˢ₀
    end
end

@testset "Regular-wave cases (Experiments 2 and 3)" begin
    # Surface Stokes drift and closed-channel return flow against Table 3 (cm/s and mm/s)
    table3 = Dict("2.A.1.1" => (1.0, 1.1), "2.A.1.2" => (2.5, 2.5), "2.A.1.3" => (4.1, 4.2),
                  "2.A.2.1" => (2.0, 1.0), "2.A.2.2" => (3.3, 1.7), "2.A.2.3" => (4.0, 2.1),
                  "2.B.1.1" => (0.62, 0.6), "2.B.1.2" => (1.8, 1.9), "2.B.1.3" => (3.7, 3.8),
                  "2.B.2.1" => (1.8, 0.9), "2.B.2.2" => (2.6, 1.4), "2.B.2.3" => (4.0, 2.1),
                  "3.A.1" => (1.1, 0.8), "3.A.2" => (2.8, 2.2), "3.B.1" => (1.1, 0.8), "3.B.2" => (2.8, 2.2))
    for name in REGULAR_WAVE_CASES
        c = anti_stokes_case(name, Float64)
        @test is_regular(c)
        @test c.family in keys(REGULAR_WAVE_FAMILIES)
        us, urf = table3[name]
        @test 1e2c.Uˢ₀ ≈ us rtol=0.08
        @test -1e3c.u_rf ≈ urf rtol=0.15
        @test c.t_FOV ≈ 8.5 / c.U₀
        @test ic_case_dirname(c) == "case_" * replace(c.family, "." => "")
    end
    @test anti_stokes_case("2A13").name == "2.A.1.3"
    @test anti_stokes_case("3.A.1", Float64).t_FOV ≈ 44.7 atol=0.1
    @test !is_regular(case_1D())
    @test ic_case_dirname(case_1D()) == "case_1D"
end

@testset "Band forcing for stationary turbulence" begin
    grid = build_grid(CPU(), Float64; Nx=32, Ny=32, Nz=8, Lx=3.2, Ly=3.2, Lz=0.8)
    mask = band_mask(grid, 2.5, 7.5)
    @test size(mask) == (32, 32, 1)
    @test mask[1, 1, 1] == 0
    @test mask[2, 1, 1] == 0                       # k = 1.96 m⁻¹ is below the band
    @test mask[3, 1, 1] == 1                       # k = 3.93 m⁻¹ is inside
    @test mask[32, 1, 1] == 0 && mask[31, 1, 1] == 1  # symmetric in ±k
    @test 20 < sum(mask) < 200
    case = anti_stokes_case("2.A.1.3", Float64)
    k_lo, k_hi = forcing_band(case)
    @test k_lo ≈ 0.5 * 0.75 / 0.15 && k_hi ≈ 1.5 * 0.75 / 0.15
    Fu, Fv = XFaceField(grid), YFaceField(grid)
    model = build_model(grid; forcing=(; u=Fu, v=Fv))
    bf = BandForcing(model; targets=(0.02, 0.02), k_lo, k_hi, γ_ref=reference_gain(case), gain=3.0, closed_loop=true)
    bf.Fu, bf.Fv = Fu, Fv
    Random.seed!(1)
    set!(model; u=(x, y, z) -> 0.02 * randn(), v=(x, y, z) -> 0.02 * randn())
    update_forcing!(bf, model)
    @test all(isfinite, interior(Fu)) && all(isfinite, interior(Fv))
    @test abs(mean(interior(Fu))) < 1e-3 * maximum(abs, interior(Fu))   # horizontal mean is not forced
    @test maximum(abs, interior(Fu)) > 0
    @test all(bf.γ .>= 0)
    # the controller: energy below target raises the gain, above target lowers it, and the integral advances
    bf.I .= 0
    set!(model; u=(x, y, z) -> 0.01 * randn(), v=(x, y, z) -> 0.01 * randn())
    update_forcing!(bf, model; Δt=1.0)
    @test all(bf.γ .> bf.γ_ref) && all(bf.I .> 0)
    bf.I .= 0
    set!(model; u=(x, y, z) -> 0.04 * randn(), v=(x, y, z) -> 0.04 * randn())
    update_forcing!(bf, model; Δt=1.0)
    @test all(bf.γ .== 0) && all(bf.I .<= 0)
    I_before = copy(bf.I)
    update_forcing!(bf, model; Δt=1.0)
    @test bf.I == I_before                      # anti-windup: clamped at zero, the integral is frozen
    # a time step with the forcing runs
    simulation = Simulation(model; Δt=0.01, stop_iteration=2, verbose=false)
    simulation.callbacks[:forcing] = Callback(forcing_callback(bf), IterationInterval(1))
    run!(simulation)
    @test length(bf.history) >= 1
    @test all(isfinite, interior(model.velocities.u))
end

@testset "Uniform group and shear profile" begin
    case = anti_stokes_case("1.D", Float64)
    packet = packet_parameters(case, 12.0, 6.0)
    p = uniform_parameters(case, packet)
    @test p.t_peak ≈ 4 * case.τ₀
    @test uniform_uˢ(0.0, p.t_peak, p) ≈ case.Uˢ₀
    @test uniform_uˢ(0.0, 0.0, p) < 1e-6 * case.Uˢ₀
    # the uniform group is the travelling packet seen from a fixed column
    for t in (p.t_peak - 1.5, p.t_peak + 2.0), z in (0.0, -0.05)
        @test uniform_uˢ(z, t, p) ≈ uˢ(6.0, 0.0, z, t, packet) rtol=1e-6
    end
    # analytic derivatives against finite differences
    δ = 1e-4
    for t in (p.t_peak - 2.0, p.t_peak + 1.0), z in (-0.01, -0.1)
        @test uniform_∂t_uˢ(z, t, p) ≈ (uniform_uˢ(z, t + δ, p) - uniform_uˢ(z, t - δ, p)) / 2δ rtol=1e-5
        @test uniform_∂z_uˢ(z, t, p) ≈ (uniform_uˢ(z + δ, t, p) - uniform_uˢ(z - δ, t, p)) / 2δ rtol=1e-5
    end
    @test shear_profile(0.0, 1.0, p) ≈ case.Uˢ₀
    @test shear_profile(-1 / (2case.k), 1.0, p) ≈ case.Uˢ₀ / ℯ
    # CL2 alignment: Eulerian and Stokes shear have the same sign for α > 0
    @test (shear_profile(-0.01, 1.0, p) - shear_profile(-0.02, 1.0, p)) * uniform_∂z_uˢ(-0.015, p.t_peak, p) > 0
    @test has_uniform_packet("sheared_packet_turbulence") && has_shear("sheared_control") && !has_packet("uniform_packet_null")
    ps = uniform_parameters(case, packet; steady=true, stokes_factor=2)
    @test uniform_uˢ(0.0, 0.0, ps) ≈ 2case.Uˢ₀ && uniform_uˢ(0.0, 100.0, ps) ≈ 2case.Uˢ₀ && uniform_∂t_uˢ(0.0, 3.0, ps) == 0
    @test is_steady("steady_sheared_turbulence") && has_shear("steady_sheared_null") && has_uniform_packet("steady_waves_null") && has_turbulence("steady_waves_turbulence")
    @test has_wind("wind_sheared_turbulence") && is_steady("wind_sheared_turbulence") && !is_steady("wind_sheared_control") && !has_uniform_packet("wind_sheared_control") && has_shear("wind_sheared_control")
    @test has_turbulence("sheared_control") && !has_turbulence("sheared_packet_null")
    # plumbing: the uniform and sheared null members run at T0 on the CPU
    root = mktempdir()
    sim_u, dir_u = run_member(; member="uniform_packet_null", level="T0", FT=Float64, arch=CPU(), root, stop_time=0.06, output_interval=0.02, progress_interval=1000)
    sim_s, dir_s = run_member(; member="sheared_packet_null", level="T0", FT=Float64, arch=CPU(), root, stop_time=0.06, output_interval=0.02, progress_interval=1000)
    @test isfile(joinpath(dir_u, "y_averages.jld2")) && isfile(joinpath(dir_s, "metadata.jld2"))
    ms = load(joinpath(dir_s, "metadata.jld2"))
    @test ms["uniform"] && ms["has_shear"] && ms["shear_amplitude"] == 1.0
    us = Array(interior(sim_s.model.velocities.u))
    @test maximum(us) > 0.8 * Float64(case.Uˢ₀)        # the shear current is present at the surface
    @test maximum(abs, Array(interior(sim_u.model.velocities.u))) < 1e-3 * Float64(case.Uˢ₀)   # uniform null stays at rest before the group
    # wind stress accelerates the surface layer: laminar wind + shear, no waves needed (control member)
    sim_w, dir_w = run_member(; member="wind_sheared_null", level="T0", FT=Float64, arch=CPU(), root, stop_time=0.06, output_interval=0.02, progress_interval=1000, wind_stress=1e-4, shear_amplitude=0.0)
    uw = Array(interior(sim_w.model.velocities.u))
    # the Lagrangian initial condition carries uˢ(z) minus its volume mean (uniform offset ≈ −6.6 mm/s),
    # and the stress accelerates the top layer: surface minus bottom exceeds most of Uˢ₀
    @test mean(uw[:, :, end]) - mean(uw[:, :, 1]) > 0.8 * Float64(case.Uˢ₀)
    mw = load(joinpath(dir_w, "metadata.jld2")); @test mw["wind_stress"] == 1e-4 && mw["has_wind"]
end

@testset "Bounded tank: single Gaussian, packet enters and leaves" begin
    case = case_1D(Float64)
    p = packet_parameters(case, 12.0, 6.0; periodic=false)
    @test p.periodic == 0
    @test p.x₀ ≈ -4case.σ₀
    @test p.x_end ≈ 12 + 4case.σ₀
    @test packet_peak_time(p) ≈ (6 + 4case.σ₀) / case.cᵍ
    @test packet_stop_time(p) ≈ (12 + 8case.σ₀) / case.cᵍ
    # no images: G is the plain Gaussian and is not periodic
    for ξ in range(-6, 6, length=13)
        @test G(ξ, p) ≈ exp(-(ξ / case.σ₀)^2)
    end
    @test abs(G(0.0, p) - G(12.0, p)) > 0.9
    # the envelope at the wall when the centre is outside is tiny at the start
    @test uˢ(0.0, 0.0, 0.0, 0.0, p) ≤ 1.2e-7 * case.Uˢ₀
    # windows and snapshots are relative to t_peak
    w = analysis_windows(packet_peak_time(p), case.τ₀)
    @test w.before[2] ≈ packet_peak_time(p) - 3case.τ₀
    @test w.after[1] ≈ packet_peak_time(p) + 3case.τ₀
    @test snapshot_times(4 * 2.8, 2.8, 22.4) ≈ [2.8, 8.4, 11.2, 14.0, 19.6, 22.4]
    # periodic parameters are unchanged
    q = packet_parameters(case, 12.0, 6.0)
    @test q.periodic == 1
    @test q.x_end ≈ 6 + 4case.σ₀
    @test packet_stop_time(q) ≈ 2packet_peak_time(q)
    # discrete divergence on a bounded grid
    grid = build_grid(CPU(), Float64; Nx=192, Ny=4, Nz=24, Lx=12, Ly=0.8, Lz=case.h, x_topology="bounded")
    uf, wf = stokes_drift_fields(grid, p, 12.0)
    fill_halo_regions!(uf); fill_halo_regions!(wf)
    div = Field(∂x(uf) + ∂z(wf)); compute!(div)
    @test maximum(abs, interior(div)) < 0.05 * 2p.k * p.Uˢ₀
end

@testset "Periodicity of the envelope" begin
    p = packet_parameters(case_1D(Float64), 12.0, 6.0)
    # Within |ξ| ≤ 5 m the omitted fourth image contributes < 1e-10; at |ξ| = Lx/2 the
    # three-image truncation error reaches ~1e-6 relative in G′′, far below Float32 precision.
    for ξ in range(-5, 5, length=61)
        @test G(ξ + p.Lx, p)   ≈ G(ξ, p)   atol=1e-7
        @test G′(ξ + p.Lx, p)  ≈ G′(ξ, p)  atol=1e-7 / p.σ₀
        @test G′′(ξ + p.Lx, p) ≈ G′′(ξ, p) atol=1e-7 / p.σ₀^2
    end
end

@testset "Analytic solenoidality and bottom condition" begin
    p = packet_parameters(case_1D(Float64), 12.0, 6.0)
    for x in range(0, 12, length=25), z in range(-0.4, 0, length=9), t in (0.0, 5.0, 11.2, 22.4)
        @test abs(∂x_uˢ(x, 0, z, t, p) + ∂z_wˢ(x, 0, z, t, p)) < 1e-14
        @test wˢ(x, 0, -p.h, t, p) == 0
    end
end

@testset "Analytic derivatives vs centered differences" begin
    p = packet_parameters(case_1D(Float64), 12.0, 6.0)
    points = [(x, z, t) for x in (0.1, 3.0, 6.0, 9.5, 11.9),
                            z in (-0.35, -0.1, -0.01, 0.0),
                            t in (0.0, 8.0, 11.2, 20.0)]

    function max_fd_error(f, ∂f, (dx, dz, dt), δ)
        err = 0.0
        for (x, z, t) in points
            fd = (f(x + δ*dx, 0, z + δ*dz, t + δ*dt, p) - f(x - δ*dx, 0, z - δ*dz, t - δ*dt, p)) / 2δ
            err = max(err, abs(fd - ∂f(x, 0, z, t, p)))
        end
        return err
    end

    checks = ((uˢ, ∂z_uˢ, (0, 1, 0), 2p.k * p.Uˢ₀),
              (uˢ, ∂t_uˢ, (0, 0, 1), p.cᵍ * p.Uˢ₀ / p.σ₀),
              (uˢ, ∂x_uˢ, (1, 0, 0), p.Uˢ₀ / p.σ₀),
              (wˢ, ∂x_wˢ, (1, 0, 0), p.Uˢ₀ / (2p.k * p.σ₀^2)),
              (wˢ, ∂t_wˢ, (0, 0, 1), p.cᵍ * p.Uˢ₀ / (2p.k * p.σ₀^2)),
              (wˢ, ∂z_wˢ, (0, 1, 0), p.Uˢ₀ / p.σ₀))

    for (f, ∂f, direction, scale) in checks
        e₁ = max_fd_error(f, ∂f, direction, 1e-3)
        e₂ = max_fd_error(f, ∂f, direction, 5e-4)
        @test e₁ < 1e-4 * scale
        @test 3 < e₁ / e₂ < 5   # second-order convergence of the centered difference
    end
end

@testset "Discrete divergence on the stretched grid" begin
    case = case_1D(Float64)
    p = packet_parameters(case, 12.0, 6.0)

    function max_divergence(Nx, Nz)
        grid = build_grid(CPU(), Float64; Nx, Ny=4, Nz, Lx=12, Ly=0.8, Lz=case.h)  # Ny ≥ halo
        uf, wf = stokes_drift_fields(grid, p, 5.0)
        fill_halo_regions!(uf)
        fill_halo_regions!(wf)
        div = Field(∂x(uf) + ∂z(wf))
        compute!(div)
        return maximum(abs, interior(div))
    end

    d₁ = max_divergence(192, 24)
    d₂ = max_divergence(384, 48)
    d₃ = max_divergence(768, 96)
    scale = 2p.k * p.Uˢ₀

    @test d₁ < 0.05 * scale
    @test d₂ < d₁ / 2.5
    @test d₃ < d₂ / 2.5
end

@testset "Float32 evaluation" begin
    p = packet_parameters(case_1D(Float32), 12.0, 6.0)
    for f in (uˢ, wˢ, ∂z_uˢ, ∂t_uˢ, ∂x_wˢ, ∂t_wˢ)
        v = f(6f0, 0f0, -0.4f0, 0f0, p)
        @test v isa Float32
        @test isfinite(v)
    end
    @test wˢ(3f0, 0f0, -0.4f0, 1f0, p) == 0
    @test uˢ(6f0, 0f0, 0f0, 11.2f0, p) ≈ p.Uˢ₀
end

@testset "Spectral turbulence generator" begin
    Nx, Ny, Nz = 96, 16, 16
    Lx, Ly, Lz = 12.0, 0.8, 0.4
    Δx, Δy, Δz = Lx / Nx, Ly / Ny, Lz / Nz
    k_cut = π / (6Δx)

    u, v, w = spectral_velocity_field(Nx, Ny, Nz, Lx, Ly, Lz; k_e=4.0, k_cut, rng=Xoshiro(1))
    @test size(u) == size(v) == size(w) == (Nx, Ny, Nz)
    @test all(isfinite, u)
    @test abs(mean(u)) < 1e-12 * std(u)
    @test abs(mean(v)) < 1e-12 * std(v)
    @test abs(mean(w)) < 1e-12 * std(w)

    # Same seed gives the same field
    u₂, _, _ = spectral_velocity_field(Nx, Ny, Nz, Lx, Ly, Lz; k_e=4.0, k_cut, rng=Xoshiro(1))
    @test u₂ == u

    # Discrete divergence on the staggered grid is small compared with the strain
    div = (circshift(u, (-1, 0, 0)) .- u) ./ Δx .+
          (circshift(v, (0, -1, 0)) .- v) ./ Δy .+
          (circshift(w, (0, 0, -1)) .- w) ./ Δz
    strain = (circshift(u, (-1, 0, 0)) .- u) ./ Δx
    @test sqrt(mean(abs2, div)) < 0.2 * sqrt(mean(abs2, strain))

    # Integral scale of a cosine: ∫₀^{λ/4} cos(2πr/λ) dr = λ / 2π
    λ = 3.0
    x = Δx .* (0:Nx-1)
    uc = repeat(cos.(2π .* x ./ λ), 1, 4, 4)
    @test streamwise_integral_scale(uc, Δx) ≈ λ / 2π rtol=0.03

    # Calibration hits the target integral scale
    L_target = 0.2
    k_e = calibrate_k_e(192, Ny, Nz, Lx, Ly, Lz; L_target, k_cut=π / (6Lx / 192), seed=2)
    u₃, _, _ = spectral_velocity_field(192, Ny, Nz, Lx, Ly, Lz; k_e, k_cut=π / (6Lx / 192), rng=Xoshiro(2))
    @test streamwise_integral_scale(u₃, Lx / 192) ≈ L_target rtol=0.05

    # Column interpolation is exact on coinciding coordinates and clamps outside
    zf = collect(range(-0.4, 0, length=9))
    f = randn(3, 2, 9)
    @test interpolate_columns(f, zf, zf) ≈ f
    @test interpolate_columns(f, zf, [-1.0])[:, :, 1] ≈ f[:, :, 1]
    @test interpolate_columns(f, zf, [0.5])[:, :, 1] ≈ f[:, :, end]
end
