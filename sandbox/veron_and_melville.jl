#####
##### An attempt to simulate a scenario similar to that reported by
##### Veron and Melville, JFM (2001)
#####

using Oceananigans
using Oceananigans.Units
using Printf

#####
##### Domain
#####

@info "Building a grid..." 

N = 128

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

β = 1e-6
@inline u_wind(x, y, t, β) = - β * sqrt(t)
u_wind_bc = FluxBoundaryCondition(u_wind, parameters=β)

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
                            boundary_conditions = (u = u_bcs_free_slip, v = v_bcs_free_slip, w = w_bcs_free_slip),
                            closure = IsotropicDiffusivity(ν=2e-6),

                            #boundary_conditions = (u = u_bcs_no_slip, v = v_bcs_no_slip, w = w_bcs_no_slip),
                            #closure = AnisotropicMinimumDissipation(),
                            
                            coriolis = nothing,
                            tracers = nothing,
                            buoyancy = nothing)

u₀ = sqrt(β * sqrt(60))

set!(model, w = (x, y, z) -> 1e0 * u₀ * exp(z / (5 * grid.Δz)) * rand())

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
                                    
wizard = TimeStepWizard(cfl=0.5, Δt=0.01, max_Δt=0.1)
simulation = Simulation(model, Δt=wizard, stop_time=10minutes, progress=progress, iteration_interval=10)

@show simulation

#####
##### Set up output
#####

prefix = @sprintf("veron_and_melville_Nz%d_β%.1e", grid.Nz, β)
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

@time run!(simulation)

@info "Simulation complete: $simulation. Output:"

for (name, writer) in simulation.output_writers
    absfilepath = abspath(writer.filepath)
    @info "OutputWriter $name, $absfilepath:\n $writer"
end
