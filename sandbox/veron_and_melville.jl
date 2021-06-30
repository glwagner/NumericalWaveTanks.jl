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

# ∂t_uˢ(z, t) =
g = 9.81
a = 0.002
const k = 360
const uˢ = a^2 * k * sqrt(g * k)
@show λ = 2π / k
∂z_uˢ(z, t) = 2k * uˢ * exp(2k * z)

β = 1e-6
@inline wind_stress(x, y, t, β) = - β * sqrt(t)
u_wind_bc = FluxBoundaryCondition(wind_stress, parameters=β)

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
                            closure = IsotropicDiffusivity(ν=1.05e-6, κ=0),

                            #boundary_conditions = (u = u_bcs_no_slip, v = v_bcs_no_slip, w = w_bcs_no_slip),
                            #closure = AnisotropicMinimumDissipation(),
                            
                            stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                            coriolis = nothing,
                            tracers = :c,
                            buoyancy = nothing)

u₀ = sqrt(1e-4 * sqrt(60))

set!(model,
     w = (x, y, z) -> 1e0 * u₀ * exp(z / (5 * grid.Δz)) * rand(),
     #c = (x, y, z) -> exp(z / 0.05),
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
                                    
wizard = TimeStepWizard(cfl=0.5, Δt=0.01, max_Δt=0.1)
simulation = Simulation(model, Δt=wizard, stop_time=4minutes, progress=progress, iteration_interval=10)

@show simulation

#####
##### Set up output
#####

prefix = @sprintf("veron_and_melville_Nz%d_β%.1e_waves", grid.Nz, β)

outputs = merge(model.velocities, model.tracers)

simulation.output_writers[:yz] = JLD2OutputWriter(model, outputs,
                                                  schedule = TimeInterval(0.1),
                                                  prefix = prefix * "_yz",
                                                  field_slicer = FieldSlicer(i = round(Int, grid.Nx/2)),
                                                  force = true)

simulation.output_writers[:xz] = JLD2OutputWriter(model, outputs,
                                                  schedule = TimeInterval(0.1),
                                                  field_slicer = FieldSlicer(j = round(Int, grid.Ny/2)),
                                                  prefix = prefix * "_xz",
                                                  force = true)

simulation.output_writers[:z] = JLD2OutputWriter(model, outputs,
                                                 schedule = TimeInterval(0.01),
                                                 field_slicer = FieldSlicer(i = round(Int, grid.Nx/2),
                                                                            j = round(Int, grid.Ny/2)),
                                                 prefix = prefix * "_z",
                                                 force = true)

@info "Running..."

@time run!(simulation)

@info "Simulation complete: $simulation. Output:"

for (name, writer) in simulation.output_writers
    absfilepath = abspath(writer.filepath)
    @info "OutputWriter $name, $absfilepath:\n $writer"
end
