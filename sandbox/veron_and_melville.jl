#####
##### An attempt to simulate a scenario similar to that reported by
##### Veron and Melville, JFM (2001)
#####

using Oceananigans
using Oceananigans.Units
using Printf
using CUDA

#####
##### Domain
#####

@info "Building a grid..." 

N = 256

grid = RegularRectilinearGrid(size = (2N, N, N), halo = (3, 3, 3), 
                              x = (0, 0.6),  # longer in streamwise direction
                              y = (0, 0.3),  # wide enough to avoid finite-width effects?
                              z = (-0.3, 0), # not full depth, but deep enough?
                              topology = (Periodic, Bounded, Bounded))

@show grid

#####
##### Boundary conditions
##### We define two options: one using drag model, one no-slip
#####

@info "Defining boundary conditions..."

# Simple model for momenum fluxes at solid walls
cᵈ = 2e-3
u_drag(x, y, t, u, v, w, cᵈ) = - cᵈ * u * sqrt(u^2 + v^2 + w^2)
v_drag(x, y, t, u, v, w, cᵈ) = - cᵈ * v * sqrt(u^2 + v^2 + w^2)
w_drag(x, y, t, u, v, w, cᵈ) = - cᵈ * w * sqrt(u^2 + v^2 + w^2)

#####
##### Stokes drift
#####

# Physical parameters
const g = 9.81 # gravitational acceleration
const T = 7.2e-5 # (kinematic) surface tension, m³ s⁻²

# Surface wave parameters
const a₀ = 0.002 # m, constant wave amplitude, or amplitude at t = τᵃ
const τᵃ = 10 # s, time-scale over which surface wave amplitude increases
const k₀ = 2π / 0.02 # m⁻¹, constant wavenumber, or wavenumber at t = τᵏ

@inline a(t) = a₀ * exp(t / τᵃ)
@inline ∂t_a(t) = a₀ * exp(t / τᵃ) / τᵃ

const ω′ = sqrt(g * k₀) # gravity frequency without surface tension effects
const ω₀ = sqrt(g * k₀ + T * k₀^3) # gravity wave frequency with surface tension

@show ω₀ ω′

# Time-varying k --- should we implement?
# const τᵏ = 10 # s
# @inline k(t) = k₀ * exp(t / τᵏ)
# @inline ω²(t) = g * k(t) + T * k(t)^3 # gravity wave frequency w/ surface tension
# @inline Uˢ(t) = a(t)^2 * k(t) * sqrt(ω²(t)) # time-dependent k

@inline    uˢ(z, t) = a(t) * a(t) * k₀ * ω₀ * exp(2k₀ * z) # constant k
@inline ∂t_uˢ(z, t) = 2 * ∂t_a(t) * uˢ(z, t) / a(t)
@inline ∂z_uˢ(z, t) = 2k₀ * uˢ(z, t)

# ∫ᶻ_∂t_uˢ(t) = ∂t_a(t) * uˢ(z=0, t) / (k₀ * a(t))
@inline ∫ᶻ_∂t_uˢ(t) = a(t) * ∂t_a(t) * ω₀

# From Veron and Melville 2001: β = α / ρ
#
# α = 10⁻² => β ≈ 10⁻⁵
#
# As a sanity check: estimate τ_dynamic ≈ ρ_air * Cᴰ * u_air²
# Then with ρ_air ≈ 1.2, Cᴰ ≈ 10⁻³, u_air = 10 => τ_dynamic ≈ 0.12

β = 1e-5 # m² s^(-5/2), from Fabrice Veron (June 30 2021 and May 11 2021)

@inline kinematic_wind_stress(x, y, t, β) = - (β * sqrt(t) - ∫ᶻ_∂t_uˢ(t))

u_wind_bc = FluxBoundaryCondition(kinematic_wind_stress, parameters=β)

u_drag_bc = FluxBoundaryCondition(u_drag, field_dependencies=(:u, :v, :w), parameters = cᵈ)
v_drag_bc = FluxBoundaryCondition(v_drag, field_dependencies=(:u, :v, :w), parameters = cᵈ)
w_drag_bc = FluxBoundaryCondition(w_drag, field_dependencies=(:u, :v, :w), parameters = cᵈ)

u_bcs_drag = UVelocityBoundaryConditions(grid, top = u_wind_bc, bottom = u_drag_bc, south = u_drag_bc, north = u_drag_bc)
v_bcs_drag = VVelocityBoundaryConditions(grid, bottom = v_drag_bc)
w_bcs_drag = WVelocityBoundaryConditions(grid, north = w_drag_bc, south = w_drag_bc)

no_slip_bc = ValueBoundaryCondition(0)
u_bcs_no_slip = UVelocityBoundaryConditions(grid, top = u_wind_bc, bottom = no_slip_bc, south = no_slip_bc, north = no_slip_bc)
v_bcs_no_slip = VVelocityBoundaryConditions(grid, bottom = no_slip_bc)
w_bcs_no_slip = WVelocityBoundaryConditions(grid, north = no_slip_bc, south = no_slip_bc)

u_bcs_free_slip = UVelocityBoundaryConditions(grid, top = u_wind_bc)
v_bcs_free_slip = VVelocityBoundaryConditions(grid)
w_bcs_free_slip = WVelocityBoundaryConditions(grid)

@info "Modeling..."

model = IncompressibleModel(architecture = GPU(),
                            advection = WENO5(),
                            timestepper = :RungeKutta3,
                            grid = grid,
                            tracers = :c,
                            # boundary_conditions = (u = u_bcs_no_slip, v = v_bcs_no_slip, w = w_bcs_no_slip),
                            boundary_conditions = (u = u_bcs_free_slip, v = v_bcs_free_slip, w = w_bcs_free_slip),
                            closure = IsotropicDiffusivity(ν=1.05e-6, κ=1.0e-6),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ, ∂t_uˢ=∂t_uˢ),
                            coriolis = nothing,
                            buoyancy = nothing)

u₀ = 1e-1 * sqrt(β * sqrt(60))

set!(model,
     w = (x, y, z) -> u₀ * exp(z / (5 * grid.Δz)) * rand(),
     # c = (x, y, z) -> exp(z / 0.05),
    )

c = model.tracers.c
CUDA.@allowscalar view(c, :, :, grid.Nz) .= 1

@info "Revvving up a simulation..."

import Oceananigans.Utils: prettytime

prettytime(s::Simulation) = prettytime(s.model.clock.time)
iteration(s::Simulation) = s.model.clock.iteration

function progress(s)

    ν = s.model.closure.ν
    wmax = maximum(abs, s.model.velocities.w)
    umax = maximum(abs, s.model.velocities.u)
    t = s.model.clock.time
    h = √(ν * t)
    Re = umax * h / ν

    @info @sprintf("Time: %s, iteration: %d, next Δt: %s, max|w|: %.2e m s⁻¹, max|u|: %.2e m s⁻¹, Re: %.2e",
                   prettytime(s), iteration(s), prettytime(s.Δt.Δt), wmax, umax, Re)

    return nothing
end
                                    
wizard = TimeStepWizard(cfl=0.7, Δt=0.01, max_Δt=1.0)
simulation = Simulation(model, Δt=wizard, stop_time=1minutes, progress=progress, iteration_interval=10)

@show simulation

#####
##### Set up output
#####

prefix = @sprintf("veron_and_melville_Nz%d_Ly%.1f_β%.1e_unsteady_waves", grid.Nz, grid.Ly, β)

outputs = merge(model.velocities, model.tracers)

simulation.output_writers[:yz] = NetCDFOutputWriter(model, outputs,
                                                    schedule = TimeInterval(0.1),
                                                    mode = "c",
                                                    filepath = prefix * "_yz.nc",
                                                    field_slicer = FieldSlicer(i = round(Int, grid.Nx/2)))

simulation.output_writers[:xz] = NetCDFOutputWriter(model, outputs,
                                                    schedule = TimeInterval(0.1),
                                                    mode = "c",
                                                    field_slicer = FieldSlicer(j = round(Int, grid.Ny/2)),
                                                    filepath = prefix * "_xz.nc")

simulation.output_writers[:z] = NetCDFOutputWriter(model, outputs,
                                                   schedule = TimeInterval(0.01),
                                                   mode = "c",
                                                   field_slicer = FieldSlicer(i = round(Int, grid.Nx/2),
                                                                              j = round(Int, grid.Ny/2)),
                                                   filepath = prefix * "_z.nc")

C = AveragedField(model.tracers.c, dims=(1, 2))
U = AveragedField(model.velocities.u, dims=(1, 2))

simulation.output_writers[:averages] = NetCDFOutputWriter(model, (c=C, u=U),
                                                          schedule = TimeInterval(0.1),
                                                          mode = "c",
                                                          filepath = prefix * "_averages.nc")

@info "Running..."

@time run!(simulation)

@info "Simulation complete: $simulation. Output:"

for (name, writer) in simulation.output_writers
    absfilepath = abspath(writer.filepath)
    @info "OutputWriter $name, $absfilepath:\n $writer"
end
