using Oceananigans

#####
##### Domain
#####

@info "Building a grid..." 

grid = RegularRectilinearGrid(size=(128, 32, 16), x=(0, 24), y=(0, 2.4), z=(-1.2, 0),
                              topology = (Bounded, Bounded, Bounded))

@show grid

#####
##### Boundary conditions
#####

@info "Defining boundary conditions..."

# Simple model for momenum fluxes at solid walls
cᵈ = 2e-3
u_drag(x, y, t, u, v, w, cᵈ) = - cᵈ * u * sqrt(u^2 + v^2 + w^2)
v_drag(x, y, t, u, v, w, cᵈ) = - cᵈ * v * sqrt(u^2 + v^2 + w^2)
w_drag(x, y, t, u, v, w, cᵈ) = - cᵈ * w * sqrt(u^2 + v^2 + w^2)

u_wind_bc = FluxBoundaryCondition(1e-4)

u_drag_bc = FluxBoundaryCondition(u_drag, field_dependencies=(:u, :v, :w), parameters = cᵈ)
v_drag_bc = FluxBoundaryCondition(v_drag, field_dependencies=(:u, :v, :w), parameters = cᵈ)
w_drag_bc = FluxBoundaryCondition(w_drag, field_dependencies=(:u, :v, :w), parameters = cᵈ)

u_bcs = UVelocityBoundaryConditions(grid, top = u_wind_bc, bottom = u_drag_bc, south = u_drag_bc, north = u_drag_bc)
v_bcs = VVelocityBoundaryConditions(grid, bottom = v_drag_bc)
w_bcs = WVelocityBoundaryConditions(grid, north = w_drag_bc, south = w_drag_bc)

#####
##### Exit sponge layer
#####

#####
##### Buoyancy
#####

buoyancy = SeawaterBuoyancy(equation_of_state = LinearEquationOfState(β = 8e-4),
                            constant_temperature = 20.0)
                                    
@info "Posing for pictures..."

model = IncompressibleModel(architecture = CPU(),
                            grid = grid,
                            coriolis = nothing,
                            tracers = nothing, #:s,
                            buoyancy = nothing, #buoyancy,
                            boundary_conditions = (u = u_bcs, v = v_bcs, w = w_bcs),
                            closure = AnisotropicMinimumDissipation())

@show model

@info "Revvving up a simulation..."

progress(s) = @info "Time: $(prettytime(s.model.clock.time))"
                                    
simulation = Simulation(model, Δt=0.01, stop_iteration=100, progress = progress)

@show simulation

@info "Flying free..."

run!(simulation)
