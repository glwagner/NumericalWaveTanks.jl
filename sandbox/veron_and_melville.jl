#####
##### An attempt to simulate a scenario similar to that reported by
##### Veron and Melville, JFM (2001)
#####

using Oceananigans

#####
##### Domain
#####

@info "Building a grid..." 

N = 32

grid = RegularRectilinearGrid(size = (4N, N, N), halo = (3, 3, 3), 
                              x = (0, 2.4),  # four times longer than width, 
                              y = (0, 0.6),  # wide enough to avoid finite-width effects
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

u_wind(x, y, t, α) = sqrt(α * t)
u_wind_bc = FluxBoundaryCondition(u_wind, parameters=1e-2 / 60)

u_drag_bc = FluxBoundaryCondition(u_drag, field_dependencies=(:u, :v, :w), parameters = cᵈ)
v_drag_bc = FluxBoundaryCondition(v_drag, field_dependencies=(:u, :v, :w), parameters = cᵈ)
w_drag_bc = FluxBoundaryCondition(w_drag, field_dependencies=(:u, :v, :w), parameters = cᵈ)

u_bcs_drag = UVelocityBoundaryConditions(grid, top = u_wind_bc, bottom = u_drag_bc, south = u_drag_bc, north = u_drag_bc)
v_bcs_drag = VVelocityBoundaryConditions(grid, bottom = v_drag_bc)
w_bcs_drag = WVelocityBoundaryConditions(grid, north = w_drag_bc, south = w_drag_bc)

u_bcs_no_slip = UVelocityBoundaryConditions(grid, top = u_wind_bc,
                                                  bottom = ValueBoundaryCondition(0),
                                                  south = ValueBoundaryCondition(0),
                                                  north = ValueBoundaryCondition(0))
v_bcs_no_slip = VVelocityBoundaryConditions(grid, bottom = ValueBoundaryCondition(0))
w_bcs_no_slip = WVelocityBoundaryConditions(grid, north = ValueBoundaryCondition(0),
                                                  south = ValueBoundaryCondition(0))

@info "Modeling..."

model = IncompressibleModel(architecture = CPU(),
                            advection = WENO5(),
                            timestepper = :RungeKutta3,
                            grid = grid,
                            coriolis = nothing,
                            tracers = nothing,
                            buoyancy = nothing,
                            boundary_conditions = (u = u_bcs_drag, v = v_bcs_drag, w = w_bcs_drag),
                            closure = IsotropicDiffusivity(ν=1e-6)) #AnisotropicMinimumDissipation())

@show model

u₀ = sqrt(1e-4)

set!(model, w = (x, y, z) -> 1e-2 * u₀ * exp(z / 0.02) * rand())

@info "Revvving up a simulation..."

import Oceananigans.Utils: prettytime
prettytime(s::Simulation) = prettytime(s.model.clock.time)
iteration(s::Simulation) = s.model.clock.iteration

progress(s) = @info "Time: $(prettytime(s)), iteration: $(iteration(s))"
                                    
wizard = TimeStepWizard(cfl=0.5, Δt=0.01) 
simulation = Simulation(model, Δt=wizard, stop_time=10, progress=progress, iteration_interval=10)

@show simulation

#####
##### Set up output
#####

simulation.output_writers[:yz] = JLD2OutputWriter(model, model.velocities,
                                                  schedule = IterationInterval(10),
                                                  prefix = "veron_melville_yz",
                                                  field_slicer = FieldSlicer(i = round(Int, grid.Nx/2)),
                                                  force = true)

simulation.output_writers[:xz] = JLD2OutputWriter(model, model.velocities,
                                                  schedule = IterationInterval(10),
                                                  field_slicer = FieldSlicer(j = round(Int, grid.Ny/2)),
                                                  prefix = "veron_melville_xz",
                                                  force = true)

@info "Flying free..."

run!(simulation)

xw, yw, zw = nodes(model.velocities.w)

using JLD2, Plots

yz_file = jldopen(simulation.output_writers[:yz].filepath)
iterations = parse.(Int, keys(yz_file["timeseries/t"]))
last_iter = iterations[end]

anim = @animate for (i, iter) in enumerate(iterations)
    wyz = yz_file["timeseries/w/$iter"][1, :, :]
    t = yz_file["timeseries/t/$iter"]

    wlim = 1e-4
    wmax = maximum(abs, wyz)
    wlevels = range(-wlim, stop=wlim, length=30)
    wlim < wmax && (levels = vcat([-wmax], levels, [wmax]))

    wyz_title = @sprintf("w(y, z, t) (m s⁻¹) at t = %s ", prettytime(t))

    contourf(yw, zw, wyz';
             linewidth = 0,
             aspectratio = :equal,
             xlims = (0, grid.Ly),
             ylims = (-grid.Lz, 0),
             color = :balance,
             levels = wlevels,
             clims = (-wlim, wlim),
             title = wyz_title)
end

gif(anim, "veron_and_melvillw.gif", fps=8)
