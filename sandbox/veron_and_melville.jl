#####
##### An attempt to simulate a scenario similar to that reported by
##### Veron and Melville, JFM (2001)
#####

using Oceananigans

#####
##### Domain
#####

@info "Building a grid..." 

N = 128

grid = RegularRectilinearGrid(size = (4N, N, N), halo = (3, 3, 3), 
                              x = (0, 1.2),  # four times longer than width, 
                              y = (0, 0.3),  # wide enough to avoid finite-width effects
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

τ₀ = 1e-4
t₀ = 60
u_wind(x, y, t, α) = - sqrt(α * t)
u_wind_bc = FluxBoundaryCondition(u_wind, parameters = τ₀^2 / t₀)

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

@info "Modeling..."

model = IncompressibleModel(architecture = GPU(),
                            advection = WENO5(),
                            timestepper = :RungeKutta3,
                            grid = grid,
                            coriolis = nothing,
                            tracers = nothing,
                            buoyancy = nothing,
                            boundary_conditions = (u = u_bcs_no_slip, v = v_bcs_no_slip, w = w_bcs_no_slip),
                            closure = IsotropicDiffusivity(ν=1e-6)) #AnisotropicMinimumDissipation())

@show model

u₀ = sqrt(τ₀)

set!(model, w = (x, y, z) -> 1e-6 * u₀ * exp(z / 0.02) * rand())

@info "Revvving up a simulation..."

import Oceananigans.Utils: prettytime
prettytime(s::Simulation) = prettytime(s.model.clock.time)
iteration(s::Simulation) = s.model.clock.iteration

progress(s) = @info "Time: $(prettytime(s)), iteration: $(iteration(s))"
                                    
wizard = TimeStepWizard(cfl=0.5, Δt=0.01) 
simulation = Simulation(model, Δt=wizard, stop_time=20, progress=progress, iteration_interval=10)

@show simulation

#####
##### Set up output
#####

prefix = "veron_and_melville_Nz$N"
simulation.output_writers[:yz] = JLD2OutputWriter(model, model.velocities,
                                                  schedule = TimeInterval(0.1),
                                                  prefix = prefix * "_yz",
                                                  field_slicer = FieldSlicer(i = round(Int, grid.Nx/2)),
                                                  force = true)

simulation.output_writers[:xz] = JLD2OutputWriter(model, model.velocities,
                                                  schedule = TimeInterval(0.1),
                                                  field_slicer = FieldSlicer(j = round(Int, grid.Ny/2)),
                                                  prefix = prefix * "_xz",
                                                  force = true)

@info "Running..."

run!(simulation)
