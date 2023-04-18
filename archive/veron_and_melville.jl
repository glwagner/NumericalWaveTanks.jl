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

Ny = 128
Ly = 0.3

Nx = Int(Ny/2)
Lx = Ly/2

Nz = 2Ny # number of points in the vertical direction
Lz = Ly # domain depth

refinement = 1.5 # controls spacing near surface (higher means finer spaced)
stretching = 8   # controls rate of stretching at bottom
h(k) = (k - 1) / Nz
ζ₀(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

arch = GPU()

#=
grid = VerticallyStretchedRectilinearGrid(architecture = arch,
                                          size = (Nx, Ny, Nz),
                                          halo = (3, 3, 3),
                                          x = (0, Lx),
                                          y = (0, Ly),
                                          z_faces = k -> Lz * (ζ₀(k) * Σ(k) - 1),
                                          topology = (Periodic, Bounded, Bounded))
=#

grid = RegularRectilinearGrid(size = (Nx, Ny, Nz),
                              halo = (3, 3, 3), 
                              x = (0, Lx),  # longer in streamwise direction
                              y = (0, Ly),  # wide enough to avoid finite-width effects?
                              z = (-Lz, 0), # not full depth, but deep enough?
                              topology = (Periodic, Bounded, Bounded))

@show grid

#####
##### Boundary conditions
##### We define two options: one using drag model, one no-slip
#####

@info "Defining boundary conditions..."

#####
##### Stokes drift
#####

# Physical parameters
const g = 9.81 # gravitational acceleration
const T = 7.2e-5 # (kinematic) surface tension, m³ s⁻²

# Surface wave parameters
#const aᵣ = 0.002 # m, reference wave amplitude at t ≈ τᵃ
const aᵣ = 0.000 # m, reference wave amplitude at t ≈ τᵃ
const τᵃ = 20 # s, time-scale over which surface wave amplitude increases
const kᵣ = 2π / 0.02 # m⁻¹, constant wavenumber
const ω′ = sqrt(g * kᵣ) # gravity frequency without surface tension effects
const ωᵣ = sqrt(g * kᵣ + T * kᵣ^3) # gravity wave frequency with surface tension

@show ϵ = aᵣ * kᵣ

@show 1 / kᵣ # surface wave decay scale
@show ωᵣ ω′

#@inline a(t) = aᵣ * tanh(t / τᵃ)
#@inline ∂t_a(t) = aᵣ / τᵃ * sech(t / τᵃ) * sech(t / τᵃ)

@inline a(t) = aᵣ
@inline ∂t_a(t) = 0.0

@inline    uˢ(z, t) = a(t) * a(t) * kᵣ * ωᵣ * exp(2kᵣ * z) # constant k
@inline ∂t_uˢ(z, t) = 2 * ∂t_a(t) * a(t) * kᵣ * ωᵣ * exp(2kᵣ * z)
@inline ∂z_uˢ(z, t) = 2kᵣ * uˢ(z, t)

# ∫ᶻ_∂t_uˢ(t) = 2 * ∂t_a(t) * a(t) * kᵣ * ωᵣ * ∫ᶻ exp(2kᵣ * z) dz = ∂t_a(t) * a(t) * ωᵣ
@inline ∫ᶻ_∂t_uˢ(t) = a(t) * ∂t_a(t) * ωᵣ

# From Veron and Melville 2001: β = α / ρ
#
# α = 10⁻² => β ≈ 10⁻⁵
#
# As a sanity check: estimate τ_dynamic ≈ ρ_air * Cᴰ * u_air²
# Then with ρ_air ≈ 1.2, Cᴰ ≈ 10⁻³, u_air = 10 => τ_dynamic ≈ 0.12

const β = 1e-5 # m² s^(-5/2), from Fabrice Veron (June 30 2021 and May 11 2021)

# u_wind_bc = FluxBoundaryCondition((x, y, t) -> - β * sqrt(t) + ∫ᶻ_∂t_uˢ(t))
u_wind_bc = FluxBoundaryCondition(-1e-4)

u_bcs_free_slip = FieldBoundaryConditions(top = u_wind_bc)
boundary_conditions = (; u = u_bcs_free_slip)

#=
# Here's a few other boundary conditions one might consider:
#   * quadratic drag (for LES)
#   * no-slip (for resolved LES or DNS)
#
# Quadratic drag model for momenum fluxes at solid walls
cᵈ = 2e-3
u_drag(x, y, t, u, v, w, cᵈ) = - cᵈ * u * sqrt(u^2 + v^2 + w^2)
v_drag(x, y, t, u, v, w, cᵈ) = - cᵈ * v * sqrt(u^2 + v^2 + w^2)
w_drag(x, y, t, u, v, w, cᵈ) = - cᵈ * w * sqrt(u^2 + v^2 + w^2)

u_drag_bc = FluxBoundaryCondition(u_drag, field_dependencies=(:u, :v, :w), parameters = cᵈ)
v_drag_bc = FluxBoundaryCondition(v_drag, field_dependencies=(:u, :v, :w), parameters = cᵈ)
w_drag_bc = FluxBoundaryCondition(w_drag, field_dependencies=(:u, :v, :w), parameters = cᵈ)

u_bcs_drag = UVelocityBoundaryConditions(grid, top = u_wind_bc, bottom = u_drag_bc, south = u_drag_bc, north = u_drag_bc)
v_bcs_drag = VVelocityBoundaryConditions(grid, bottom = v_drag_bc)
w_bcs_drag = WVelocityBoundaryConditions(grid, north = w_drag_bc, south = w_drag_bc)
boundary_conditions = (u = u_bcs_drag, v = v_bcs_drag, w = w_bcs_drag)

no_slip_bc = ValueBoundaryCondition(0)
u_bcs_no_slip = UVelocityBoundaryConditions(grid, top = u_wind_bc, bottom = no_slip_bc, south = no_slip_bc, north = no_slip_bc)
v_bcs_no_slip = VVelocityBoundaryConditions(grid, bottom = no_slip_bc)
w_bcs_no_slip = WVelocityBoundaryConditions(grid, north = no_slip_bc, south = no_slip_bc)
boundary_conditions = (u = u_bcs_no_slip, v = v_bcs_no_slip, w = w_bcs_no_slip)
=#

@info "Modeling..."

addzero(args...) = 0

model = NonhydrostaticModel(architecture = arch,
                            advection = WENO5(),
                            timestepper = :RungeKutta3,
                            grid = grid,
                            tracers = :c,
                            boundary_conditions = boundary_conditions,
                            closure = IsotropicDiffusivity(ν=1.05e-6, κ=1.0e-6),
                            #stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ, ∂t_uˢ=∂t_uˢ),
                            coriolis = nothing,
                            buoyancy = nothing)

#@show Δz = minimum(parent(grid.Δzᵃᵃᶜ)) # for a stretched grid
@show Δz = grid.Δz # for a regular grid
@show uᵢ = 1e-15
#wᵢ(x, y, z) = uᵢ * exp(z / (5 * Δz)) * randn()
wᵢ(x, y, z) = uᵢ * randn()

set!(model, w = wᵢ)

c = model.tracers.c
interior(c)[:, :, grid.Nz] .= 1

@info "Revvving up a simulation..."
                                    
simulation = Simulation(model, Δt=0.01, stop_time=1minutes)

wizard = TimeStepWizard(cfl=0.8, max_Δt=1.0)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

function progress(s)

    ν = s.model.closure.ν

    umax = maximum(abs, s.model.velocities.u)
    vmax = maximum(abs, s.model.velocities.v)
    wmax = maximum(abs, s.model.velocities.w)

    t = s.model.clock.time
    h = √(ν * t)
    Re = umax * h / ν

    @info @sprintf("Time: %s, iteration: %d, next Δt: %s, max|U|: (%.2e, %.2e, %.2e)  m s⁻¹, hk: %.2f, Re: %.1f",
                   prettytime(s),
                   iteration(s),
                   prettytime(s.Δt),
                   umax, vmax, wmax,
                   h * kᵣ,
                   Re)

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

@show simulation

#####
##### Set up output
#####

prefix = @sprintf("veron_and_melville_Nz%d_Ly%.1f_β%.1e_a%.1e", grid.Nz, grid.Ly, β, aᵣ)

outputs = merge(model.velocities, model.tracers)

simulation.output_writers[:yz] = JLD2OutputWriter(model, outputs,
                                                    schedule = TimeInterval(0.1),
                                                    force = true,
                                                    prefix = prefix * "_yz",
                                                    field_slicer = FieldSlicer(i = round(Int, grid.Nx/2)))

simulation.output_writers[:xz] = JLD2OutputWriter(model, outputs,
                                                  schedule = TimeInterval(0.1),
                                                  force = true,
                                                  field_slicer = FieldSlicer(j = round(Int, grid.Ny/2)),
                                                  prefix = prefix * "_xz")

simulation.output_writers[:netcdf_xz] = NetCDFOutputWriter(model, outputs,
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
