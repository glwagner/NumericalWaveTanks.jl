#####
##### An attempt to simulate a scenario similar to that reported by
##### Veron and Melville, JFM (2001)
#####

using Oceananigans
using Oceananigans.Units
using Printf
using SpecialFunctions
using GLMakie

#####
##### Domain
#####

@info "Building a grid..." 

Nx = 128

grid = RegularRectilinearGrid(size = (2Nx, Nx),
                              halo = (3, 3), 
                              y = (    0, 0.1),
                              z = (-0.05,   0),
                              topology = (Flat, Periodic, Bounded))

@show grid

#####
##### Stokes drift
#####

# Physical parameters
g = 9.81 # gravitational acceleration
T = 7.2e-5 # (kinematic) surface tension, m³ s⁻²

const k = 2π / 0.02
const ϵ = 0.3
const ω = sqrt(g * k + T * k^3)

@show a = ϵ / k
@inline uˢ(z, t) = ϵ^2 * ω / k * exp(2k * z)
@inline ∂z_uˢ(z, t) = 2 * ϵ^2 * ω * exp(2k * z)

surface_uˢ = ϵ^2 * ω / k
surface_∂z_uˢ = 2 * ϵ^2 * ω
@show surface_uˢ
@show 1 / surface_∂z_uˢ

#####
##### Basic state
#####

closure = IsotropicDiffusivity(ν=1.05e-6, κ=1.0e-6)

Re = 1e5
ν = closure.ν
t₀ = 12

hν = sqrt(ν * t₀)
const h = 1 / k
const U₀ = ν * Re / h # Re = U L / ν

@show Sh = surface_∂z_uˢ / (U₀ / h)
@show Re h hν U₀

#@inline shear_flow(x, y, z, t) = U₀ * erfc(z / h)
@inline shear_flow(x, y, z, t) = uˢ(z, t)
U = BackgroundField(shear_flow)

model = NonhydrostaticModel(architecture = CPU(),
                            advection = CenteredSecondOrder(),
                            timestepper = :RungeKutta3,
                            grid = grid,
                            background_fields = (; u=U),
                            closure = closure,
                            stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                            coriolis = nothing,
                            tracers = nothing,
                            buoyancy = nothing)

@show Δt = 5e-1 * grid.Δy / surface_uˢ

u, v, w = model.velocities
ξ = ComputedField(∂z(v) - ∂y(w))
KE_op = @at (Center, Center, Center) 1/2 * (u^2 + v^2 + w^2)
KE = AveragedField(KE_op, dims=(1, 2, 3))

function progress(sim)
    compute!(KE)

    @info @sprintf("Iteration: %d, time: %.2e, perturbation KE: %.2e",
                   sim.model.clock.iteration,
                   sim.model.clock.time,
                   KE[1, 1, 1])

    return nothing
end

simulation = Simulation(model, Δt=Δt, stop_time=10/surface_∂z_uˢ, iteration_interval=10, progress=progress)

function grow_instability!(simulation, energy)
    ## Initialize
    simulation.model.clock.iteration = 0
    t₀ = simulation.model.clock.time = 0
    compute!(energy)
    energy₀ = energy[1, 1, 1]

    ## Grow
    run!(simulation)

    ## Analyze
    compute!(energy)
    energy₁ = energy[1, 1, 1]
    Δτ = simulation.model.clock.time - t₀

    ## ½(u² + v²) ~ exp(2 σ Δτ)
    σ = growth_rate = log(energy₁ / energy₀) / 2Δτ

    return growth_rate
end

function rescale!(model, energy; target_kinetic_energy=1e-6)
    compute!(energy)
    rescale_factor = √(target_kinetic_energy / energy[1, 1, 1])
    [parent(q) .*= rescale_factor for q in model.velocities]

    ν = model.closure.ν
    U = maximum(abs, model.velocities.v)
    L = 1 / k
    Re = U * L / ν
    @info @sprintf("Reynolds number after rescaling: %.2e", Re)

    return nothing
end

convergence(σ) = length(σ) > 1 ? abs((σ[end] - σ[end-1]) / σ[end]) : 9.1e18 # pretty big (not Inf tho)

function estimate_growth_rate(simulation, energy, ξ;
                              target_kinetic_energy = 1e-6,
                              convergence_criterion = 1e-2,
                              maximum_iterations = 100)

    σ, power_method_data = [], []

    compute!(ξ)
    push!(power_method_data, (; ξ=collect(interior(ξ)[:, 1, :])))
    iterations = 0

    while convergence(σ) > convergence_criterion && iterations < maximum_iterations
        compute!(energy)

        @info @sprintf("About to start power method iteration %d; kinetic energy: %.2e", length(σ)+1, energy[1, 1, 1])

        push!(σ, grow_instability!(simulation, energy))
        compute!(energy)

        @info @sprintf("Power method iteration %d, kinetic energy: %.2e, σⁿ: %.2e, relative Δσ: %.2e",
                       length(σ), energy[1, 1, 1], σ[end], convergence(σ))

        compute!(ξ)
        rescale!(simulation.model, energy; target_kinetic_energy)
        push!(power_method_data, (; ξ=collect(interior(ξ)[:, 1, :])))

        iterations += 1
    end

    return σ, power_method_data
end

#####
##### Estimate growth rate
#####

Re′ = 1e-2 # perturbation Re
@show KE′ = (Re′ * ν / h)^2

noise(x, y, z) = randn()
set!(model, u=noise, v=noise, w=noise)
rescale!(model, KE, target_kinetic_energy=KE′)

growth_rates, power_method_data = estimate_growth_rate(simulation, KE, ξ, target_kinetic_energy=KE′)
@info "Power iterations converged! Estimated growth rate: $(growth_rates[end])"

simulation.model.clock.time = 0.0
simulation.model.clock.iteration = 0
simulation.stop_time = 50 / surface_∂z_uˢ

# Re = U h / ν => U² = (ν * Re / h)²
rescale!(model, KE, target_kinetic_energy=KE′)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, (; ξ)),
                                                      schedule = IterationInterval(10),
                                                      prefix = "craik_leibovich_instability",
                                                      field_slicer = nothing,
                                                      force = true)

run!(simulation)

#####
##### Visualize
#####

filepath = "craik_leibovich_instability.jld2"

vt = FieldTimeSeries(filepath, "v")
wt = FieldTimeSeries(filepath, "w")
ξt = FieldTimeSeries(filepath, "ξ")

times = vt.times
Nt = length(times)

vn(n) = Array(interior(vt[n])[1, :, :])
wn(n) = Array(interior(wt[n])[1, :, :])
ξn(n) = Array(interior(ξt[n])[1, :, :])

n = Node(1)

max_v = + maximum(abs, vn(Nt))
max_w = + maximum(abs, wn(Nt))
max_ξ = + maximum(abs, ξn(Nt))

min_v = - max_v
min_w = - max_w
min_ξ = - max_ξ

v = @lift vn($n)
w = @lift wn($n)
ξ = @lift ξn($n)

fig = Figure(resolution = (1200, 1200))

ax = Axis(fig[1, 1], title="v")
hm = heatmap!(ax, v, colorrange=(min_v, max_v), colormap=:balance)
cb = Colorbar(fig[1, 2], hm)

ax = Axis(fig[2, 1], title="w")
hm = heatmap!(ax, w, colorrange=(min_w, max_w), colormap=:balance)
cb = Colorbar(fig[2, 2], hm)

ax = Axis(fig[3, 1], title="∂z(v) - ∂y(w)")
hm = heatmap!(ax, ξ, colorrange=(min_ξ, max_ξ), colormap=:balance)
cb = Colorbar(fig[3, 2], hm)

title_str = @lift "Langmuir instability at t = " * prettytime(times[$n])
ax_t = fig[0, :] = Label(fig, title_str)

record(fig, "craik_leibovich.mp4", 1:Nt, framerate=8) do i
    @info "Plotting iteration $i of $Nt..."
    n[] = i
end

display(fig)

