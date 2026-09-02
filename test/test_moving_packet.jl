using Test
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Statistics
using Random

include(joinpath(@__DIR__, "..", "experiments", "anti_stokes", "generate_turbulence.jl"))

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
