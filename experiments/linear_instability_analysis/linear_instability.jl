using Oceananigans
using SpecialFunctions
using Printf
# using GLMakie
using JLD2
using CUDA

@inline function mean_velocity(x, y, z, t, p)
    t′ = ifelse(p.time_dependent, p.t₀ + t, p.t₀)
    h = √(2 * p.ν * t′)
    U₀ = p.A * t′
    δ = z / h
    return U₀ * ((1 + δ^2) * erfc(-δ / √2) + δ * √(2/π) * exp(-δ^2 / 2))
end

struct StokesShear
    S₀ :: Float64
    k :: Float64
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
    energy₀ = CUDA.@allowscalar first(energy)

    # Grow
    run!(simulation)

    # Analyze
    compute!(energy)
    energy₁ = CUDA.@allowscalar first(energy)
    Δτ = simulation.model.clock.time - t₀

    # ½(u² + v²) ~ exp(2 σ Δτ)
    σ = growth_rate = log(energy₁ / energy₀) / 2Δτ

    return growth_rate
end

function rescale!(model, energy; target_kinetic_energy = 1e-8)
    compute!(energy)
    rescale_factor = CUDA.@allowscalar √(target_kinetic_energy / first(energy))

    for f in merge(model.velocities, model.tracers)
        f .*= rescale_factor
    end

    return nothing
end

convergence(σ) = (length(σ) < 2 || isnan(σ[end]) || isnan(σ[end-1])) ? 9.1e18 : abs((σ[end] - σ[end-1]) / σ[end])
    
function simulate_linear_growth(simulation, energy; target_kinetic_energy=1e-8, convergence_criterion=2e-6)
    # Initialize
    compute!(energy)
    rescale!(simulation.model, energy; target_kinetic_energy)

    # Estimate time-step
    u★ = sqrt(target_kinetic_energy)
    ν = simulation.model.closure.ν
    grid = simulation.model.grid
    Δ = min(minimum(parent(grid.Δzᵃᵃᶜ)), grid.Δxᶜᵃᵃ)
    adv_Δt = 0.1 * Δ / u★
    diff_Δt = 0.1 * Δ^2 / ν
    Δt = min(adv_Δt, diff_Δt)  

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

        msg = CUDA.@allowscalar begin
                @sprintf("Power method iteration %d, kinetic energy: %.2e, σⁿ: %.2e, relative Δσ: %.2e",
                          length(σ), first(energy), σ[end], convergence(σ))
        end

        @info msg

        rescale!(simulation.model, energy; target_kinetic_energy)

        # Check stability after a fixed number of iterations, quit if it's stable
        if length(σ) > 40
            stable = isnan(σ[end])
        end
    end

    # Clean up 
    σ[isnan.(σ)] .= 0

    return σ
end

function langmuir_instability_simulation(arch;
                                         t₀ = 16.0,
                                         β = 1.2e-5,
                                         Ny = 768,
                                         Nz = 512,
                                         Ly = 0.1,
                                         Lz = 0.05,
                                         ϵ = 0.08,
                                         ν = 1.05e-6,
                                         time_dependent = false,
                                         stop_time = 0.05)

    @show ϵ
    A = β * sqrt(π / 4ν)
    U = BackgroundField(mean_velocity, parameters=(; A, ν, t₀, time_dependent))

    refinement = 1.5 # controls spacing near surface (higher means finer spaced)
    stretching = 8   # controls rate of stretching at bottom
    h(k) = (k - 1) / Nz
    ζ₀(k) = 1 + (h(k) - 1) / refinement
    Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

    @info "Building a grid..." 
    # grid = RectilinearGrid(size=(Ny, Nz), y=(0, Ly), z=(-Lz, 0), topology=(Flat, Periodic, Bounded))
    grid = RectilinearGrid(arch,
                           size = (Ny, Nz),
                           halo = (3, 3),
                           y = (0, Ly),
                           z = k -> Lz * (ζ₀(k) * Σ(k) - 1), # (-Lz, 0)
                           topology = (Flat, Periodic, Bounded))

    closure = ScalarDiffusivity(; ν)
    stokes_drift = UniformStokesDrift(; ∂z_uˢ=StokesShear(; ϵ))

    model = NonhydrostaticModel(; grid, closure, stokes_drift,
                                timestepper = :RungeKutta3,
                                advection = CenteredSecondOrder(),
                                background_fields = (; u=U))

    uᵢ(x, y, z) = 1e-6 * randn()
    set!(model, u=uᵢ, v=uᵢ, w=uᵢ)
    simulation = Simulation(model; Δt=1, stop_time, verbose=false)

    return simulation
end

function estimate_growth_rate(arch; target_kinetic_energy=1e-10, kw...)
    simulation = langmuir_instability_simulation(arch; kw...)
    u, v, w = simulation.model.velocities
    e = Field((u^2 + v^2 + w^2) / 2)
    E = Field(Average(e))
    growth_rates = simulate_linear_growth(simulation, E; target_kinetic_energy)
    return growth_rates[end], simulation
end

arch = GPU()
all_time_independent_growth_rates = Dict()
inception_times = [16.0] #collect(15.0:0.5:17.0)

for ϵ = 0.01:0.01:0.30
    time_independent_growth_rates = Float64[]

    for t₀ in inception_times
        # Time-independent problem
        growth_rate, simulation = estimate_growth_rate(arch; t₀, ϵ, time_dependent=false)
        push!(time_independent_growth_rates, growth_rate)

        u, v, w = simulation.model.velocities
    
        #=
        wn = interior(w, 1, :, :)
        fig = Figure(resolution=(1200, 600))
        title = @sprintf("ϵ = %.2f, t★ = %.1f, σ = %.4f", ϵ, t₀, growth_rate)
        ax = Axis(fig[1, 1]; title, aspect=2)
        heatmap!(ax, wn)
        plotname = @sprintf("linearly_unstable_mode_t%02d_ep%02d.png", 10t₀, 100ϵ)
        save(plotname, fig)
        =#

        grid = simulation.model.grid
        Nx, Ny, Nz = size(grid)

        filename = @sprintf("linearly_unstable_mode_t0%02d_ep%02d_N%d_%d_L%d_%d.jld2",
                            10t₀, 100ϵ, Ny, Nz,
                            100 * grid.Ly,
                            100 * grid.Lz)

        rm(filename, force=true)
        file = jldopen(filename, "a+")
        file["inception_time"] = t₀
        file["growth_rate"] = growth_rate
        file["u"] = Array(parent(u))
        file["v"] = Array(parent(v))
        file["w"] = Array(parent(w))
        file["grid"] = simulation.model.grid
        close(file)

        @info """\n
              Power method growth rate estimate:
                  - ϵ: $ϵ
                  - Inception time: $t₀.
                  - Estimated growth rate (fixed mean flow): $(time_independent_growth_rates[end])
              """
    end

    #=
    all_time_independent_growth_rates[ϵ] = time_independent_growth_rates
    filename = @sprintf("linear_stability_analysis_ep%02d", 100ϵ)
    filepath = filename * ".jld2"
    file = jldopen(filepath, "a+")
    file["inception_times"] = inception_times
    file["time_independent_growth_rates"] = time_independent_growth_rates
    close(file)
    =#
end

