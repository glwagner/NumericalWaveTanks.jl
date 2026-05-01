using CUDA
using Random
using Statistics
using JLD2
using Oceananigans
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.Units
using Printf

# Mirrors constant_waves.jl's API surface but standalone (no eigenmode IC dependency).
# Goal: confirm the v0.107.4 update doesn't break the constructors and run loop we use.

struct ConstantStokesShear{T}
    a :: T
    k :: T
    ω :: T
end

ConstantStokesShear(a, k; g=9.81, γ=7.2e-5) = ConstantStokesShear{Float64}(a, k, sqrt(g * k + γ * k^3))
@inline (sh::ConstantStokesShear)(z, t) = 2 * sh.a^2 * sh.k^2 * sh.ω * exp(2 * sh.k * z)

@inline ν_∂z²_uˢ(x, y, z, t, p) = - 4 * p.ν * p.∂z_uˢ.a^2 * p.∂z_uˢ.k^3 * p.∂z_uˢ.ω * exp(2 * p.∂z_uˢ.k * z)

arch = GPU()

Nx = Ny = 64
Nz = 32
Lx = Ly = 0.2
Lz = 0.1
ϵ = 0.1
α = 1.2e-5
ν = 1.05e-6
κ = 1e-7
k = 2π / 0.03
W′ = 1e-4

refinement = 1.5
stretching = 8
h(k) = (k - 1) / Nz
ζ₀(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

@info "Building grid"
grid = RectilinearGrid(arch,
                       size = (Nx, Ny, Nz),
                       halo = (3, 3, 3),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = k -> Lz * (ζ₀(k) * Σ(k) - 1),
                       topology = (Periodic, Periodic, Bounded))
@show grid

@inline τʷ(x, y, t, p) = - p.α * sqrt(t)
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τʷ, parameters = (; α)))

a = ϵ / k
∂z_uˢ = ConstantStokesShear(a, k)
u_forcing = Forcing(ν_∂z²_uˢ, parameters = (; ∂z_uˢ, ν))

vitd = VerticallyImplicitTimeDiscretization()

@info "Building model"
model = NonhydrostaticModel(grid;
                            boundary_conditions = (; u = u_bcs),
                            advection = Centered(order=2),
                            timestepper = :RungeKutta3,
                            tracers = :c,
                            closure = ScalarDiffusivity(vitd; ν, κ),
                            stokes_drift = UniformStokesDrift(; ∂z_uˢ),
                            forcing = (; u = u_forcing))

Random.seed!(42)
uᵢ(x, y, z) = 10W′ * randn()
set!(model, u=uᵢ, v=uᵢ, w=uᵢ)

c = model.tracers.c
view(interior(c), :, :, grid.Nz) .= 1

model.clock.time = 16.0

@info "Building simulation"
simulation = Simulation(model; Δt=1e-4, stop_iteration=200)

Δ = grid.Δxᶜᵃᵃ
max_Δt = 0.1 * Δ^2 / ν
wizard = TimeStepWizard(; cfl=1.0, max_Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))

wall_clock = Ref(time_ns())
function progress(sim)
    umax = maximum(abs, sim.model.velocities.u)
    vmax = maximum(abs, sim.model.velocities.v)
    wmax = maximum(abs, sim.model.velocities.w)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    @info @sprintf("iter: %d, t: %s, Δt: %s, wall: %s, max|U|=(%.2e, %.2e, %.2e)",
                   iteration(sim), prettytime(time(sim)), prettytime(sim.Δt),
                   prettytime(elapsed), umax, vmax, wmax)
    wall_clock[] = time_ns()
    return nothing
end
simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

dir = "smoke_test_output"
outputs = merge(model.velocities, model.tracers)
U = Field(Average(model.velocities.u, dims=(1, 2)))
C = Field(Average(model.tracers.c, dims=(1, 2)))

simulation.output_writers[:avg] = JLD2Writer(model, (u=U, c=C); dir,
    schedule = TimeInterval(0.05),
    filename = "smoke_averages",
    overwrite_existing = true)

simulation.output_writers[:slice] = JLD2Writer(model, outputs; dir,
    schedule = TimeInterval(0.05),
    filename = "smoke_xz_slice",
    indices = (:, 1, :),
    overwrite_existing = true)

@info "Running simulation"
run!(simulation)

@info "Smoke test complete: ran $(iteration(simulation)) iterations to t=$(time(simulation))"

# Sanity: any NaNs?
u, v, w = model.velocities
c_field = model.tracers.c
for (name, fld) in (:u=>u, :v=>v, :w=>w, :c=>c_field)
    arr = Array(interior(fld))
    if any(isnan, arr)
        error("NaN detected in $name")
    end
    @info @sprintf("%s: min=%.3e max=%.3e", name, minimum(arr), maximum(arr))
end

@info "All checks passed"
