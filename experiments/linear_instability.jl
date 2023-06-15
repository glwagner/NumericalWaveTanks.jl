using Oceananigans
using SpecialFunctions
using Printf
using GLMakie

struct StokesShear
    S₀ :: Float64
    k :: Float64
end

@inline function mean_velocity(x, y, z, t, p)
    t′ = ifelse(p.time_dependent, p.t₀ + t, p.t₀)
    h = √(2 * p.ν * t′)
    U₀ = p.A * t′
    δ = z / h

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

convergence(σ) = (length(σ) < 2 || isnan(σ[end]) || isnan(σ[end-1])) ? 9.1e18 : abs((σ[end] - σ[end-1]) / σ[end])
    
function simulate_linear_growth(simulation, energy; target_kinetic_energy=1e-8, convergence_criterion=2e-3)
    # Initialize
    compute!(energy)
    rescale!(simulation.model, energy; target_kinetic_energy)

    # Estimate time-step
    u★ = sqrt(target_kinetic_energy)
    grid = simulation.model.grid
    Δx = min(grid.Ly / grid.Ny, grid.Lz / grid.Nz)
    Δt = 2e-4 * Δx / u★
    simulation.Δt = Δt
    @info "Setting time step to $Δt, estimated iterations: " *
          string(ceil(Int, simulation.stop_time / simulation.Δt))

    iterating = true
    stable = false
    σ = Float64[]
    while !stable && (convergence(σ) > convergence_criterion)
        compute!(energy)

        # Estimate growth rate
        σⁿ★ = grow_instability!(simulation, energy)

        # If "growth rate" is negative, set to NaN (the setup is stable)
        σⁿ = ifelse(σⁿ★ < 0, NaN, σⁿ★)
        push!(σ, σⁿ)

        compute!(energy)

        @info @sprintf("Power method iteration %d, kinetic energy: %.2e, σⁿ: %.2e, relative Δσ: %.2e",
                       length(σ), energy[1, 1, 1], σ[end], convergence(σ))

        rescale!(simulation.model, energy; target_kinetic_energy)

        # Check stability after a fixed number of iterations, quit if it's stable
        if length(σ) > 20
            stable = isnan(σ[end])
        end
    end

    # Clean up 
    σ[isnan.(σ)] .= 0

    return σ
end

function langmuir_instability_simulation(; t₀ = 15.0,
                                           A = 1e-2,
                                           Ny = 256,
                                           Nz = 128,
                                           Ly = 0.06,
                                           Lz = 0.02,
                                           ν = 1.05e-6,
                                           time_dependent = true,
                                           stop_time = 0.1)

    U = BackgroundField(mean_velocity, parameters=(; A, ν, t₀, time_dependent))
    grid = RectilinearGrid(size=(Ny, Nz), y=(0, Ly), z=(-Lz, 0), topology=(Flat, Periodic, Bounded))
    closure = ScalarDiffusivity(; ν)
    stokes_drift = UniformStokesDrift(; ∂z_uˢ=StokesShear())

    model = NonhydrostaticModel(; grid, closure, stokes_drift,
                                timestepper = :RungeKutta3,
                                advection = CenteredSecondOrder(),
                                background_fields = (; u=U))

    uᵢ(x, y, z) = 1e-6 * randn()
    set!(model, u=uᵢ, v=uᵢ, w=uᵢ)
    simulation = Simulation(model; Δt=1, stop_time, verbose=false)

    return simulation
end

function estimate_growth_rate(; target_kinetic_energy=1e-10, kw...)
    simulation = langmuir_instability_simulation(; kw...)
    u, v, w = simulation.model.velocities
    e = Field((u^2 + v^2 + w^2) / 2)
    E = Field(Average(e))
    growth_rates = simulate_linear_growth(simulation, E; target_kinetic_energy)
    return growth_rates[end], simulation
end

inception_times = collect(1.0:2.0:20.0)
time_independent_growth_rates = Float64[]
time_dependent_growth_rates = Float64[]
strong_time_dependent_growth_rates = Float64[]

for t₀ in inception_times
    # Time-independent problem
    growth_rate, simulation = estimate_growth_rate(; t₀, time_dependent=false)
    push!(time_independent_growth_rates, growth_rate)

    u, v, w = simulation.model.velocities
    wn = interior(w, 1, :, :)
    fig = Figure(resolution=(1200, 600))
    ax = Axis(fig[1, 1], aspect=2)
    heatmap!(ax, wn)
    display(current_figure())

    # Time-dependent problem
    growth_rate, simulation = estimate_growth_rate(; t₀, time_dependent=true)
    push!(time_dependent_growth_rates, growth_rate)

    u, v, w = simulation.model.velocities
    wn = interior(w, 1, :, :)
    fig = Figure(resolution=(1200, 600))
    ax = Axis(fig[1, 1], aspect=2)
    heatmap!(ax, wn)
    display(current_figure())

    # Time-dependent problem
    growth_rate, simulation = estimate_growth_rate(; t₀, target_kinetic_energy=1e-8, time_dependent=true)
    push!(strong_time_dependent_growth_rates, growth_rate)

    @info """\n
          Power method growth rate estimate:
              - Inception time: $t₀.
              - Estimated growth rate (fixed mean flow):                                 $(time_independent_growth_rates[end])
              - Estimated growth rate (time-dependent mean flow):                        $(time_dependent_growth_rates[end])
              - Estimated growth rate (time-dependent mean flow, stronger perturbation): $(strong_time_dependent_growth_rates[end])

          """
end

# Save data
using JLD2

filename = "linear_instability_analysis"
filepath = filename * ".jld2"

n = 1
while isfile(filepath)
    global filepath = filename * "_$(n).jld2"
    global n += 1
end

file = jldopen(filepath, "a+")
file["inception_times"] = inception_times
file["time_independent_growth_rates"] = time_independent_growth_rates
file["time_dependent_growth_rates"] = time_dependent_growth_rates
file["strong_time_dependent_growth_rates"] = strong_time_dependent_growth_rates
close(file)

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Inception time (s)", ylabel="Growth rate (s⁻¹)")
scatter!(ax, inception_times, time_independent_growth_rates, label="Fixed base flow")
scatter!(ax, inception_times, time_dependent_growth_rates, label="Accelerating base flow")
display(fig)

