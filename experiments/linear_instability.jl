using Oceananigans
using SpecialFunctions
using Printf

struct StokesShear
    S₀ :: Float64
    k :: Float64
end

@inline function mean_velocity(x, y, z, t, p)
    U₀ = p.U₀
    δ = z / p.h
    return U₀ * ((1 + δ^2) * erfc(-δ / √2) + δ * √(2/π) * exp(-δ^2 / 2))
end

function StokesShear(; g=9.81, T=7.2e-5, k=2π/0.03, ϵ=0.14)
    ω = sqrt(g * k + T * k^3)
    S₀ = 2 * ω * ϵ^2
    return StokesShear(S₀, k)
end

@inline (S::StokesShear)(z, t) = S.S₀ * exp(2 * S.k * z)

function grow_instability!(simulation, energy)
    # Initialize
    simulation.model.clock.iteration = 0
    t₀ = simulation.model.clock.time = 0
    compute!(energy)
    energy₀ = energy[1, 1, 1]

    # Grow
    run!(simulation)

    # Analyze
    compute!(energy)
    energy₁ = first(energy)
    Δτ = simulation.model.clock.time - t₀

    # ½(u² + v²) ~ exp(2 σ Δτ)
    σ = growth_rate = log(energy₁ / energy₀) / 2Δτ

    return growth_rate
end

function rescale!(model, energy; target_kinetic_energy = 1e-8)
    compute!(energy)
    rescale_factor = √(target_kinetic_energy / first(energy))

    for f in merge(model.velocities, model.tracers)
        f .*= rescale_factor
    end

    return nothing
end

convergence(σ) = length(σ) > 1 ? abs((σ[end] - σ[end-1]) / σ[end]) : 9.1e18 # pretty big (not Inf tho)

function simulate_linear_growth(simulation, energy; target_kinetic_energy=1e-8, convergence_criterion=5e-2)
    σ = []

    while convergence(σ) > convergence_criterion
        compute!(energy)

        @info @sprintf("About to start power method iteration %d; kinetic energy: %.2e", length(σ)+1, first(energy))
        push!(σ, grow_instability!(simulation, energy))
        compute!(energy)

        @info @sprintf("Power method iteration %d, kinetic energy: %.2e, σⁿ: %.2e, relative Δσ: %.2e",
                       length(σ), energy[1, 1, 1], σ[end], convergence(σ))

        rescale!(simulation.model, energy; target_kinetic_energy)
    end

    return σ
end

function langmuir_instability_simulation(; t = 15.0,
                                           A = 1e-2,
                                           Ny = 128,
                                           Nz = 64,
                                           Ly = 0.05,
                                           Lz = 0.025,
                                           ν = 1.05e-6,
                                           Δt = 2e-3,
                                           stop_iteration = 100)
    U₀ = A * t
    h = sqrt(ν * t)
    U = BackgroundField(mean_velocity, parameters=(; U₀, h))
    grid = RectilinearGrid(size=(Ny, Nz), y=(0, Ly), z=(-Lz, 0), topology=(Flat, Periodic, Bounded))
    closure = ScalarDiffusivity(; ν)
    stokes_drift = UniformStokesDrift(; ∂z_uˢ=StokesShear())

    model = NonhydrostaticModel(; grid, closure, stokes_drift,
                                timestepper = :RungeKutta3,
                                advection = CenteredSecondOrder(),
                                background_fields = (; u=U))

    uᵢ(x, y, z) = 1e-1 * U₀ * randn()
    set!(model, u=uᵢ, v=uᵢ, w=uᵢ)
    simulation = Simulation(model; Δt, stop_iteration, verbose=false)

    return simulation
end

function estimate_growth_rate(; t=1.0, kw...)
    simulation = langmuir_instability_simulation(t=1.0, kw...)
    u, v, w = simulation.model.velocities
    energy = Field(Average((u^2 + v^2 + w^2) / 2))
    target_kinetic_energy = 1e-8
    rescale!(simulation.model, energy; target_kinetic_energy)
    growth_rates = simulate_linear_growth(simulation, energy; target_kinetic_energy)

    return growth_rates[end]
end

times = 1.0:1.0:20.0
growth_rates = []
for t in times
    growth_rate = estimate_growth_rate(; t)
    push!(growth_rates, growth_rate)
end

