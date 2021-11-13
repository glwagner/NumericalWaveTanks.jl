using Oceananigans
using Oceananigans.Units
using Printf

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

grid = VerticallyStretchedRectilinearGrid(architecture = arch,
                                          size = (Nx, Ny, Nz),
                                          halo = (3, 3, 3),
                                          x = (0, Lx),
                                          y = (0, Ly),
                                          z_faces = k -> Lz * (ζ₀(k) * Σ(k) - 1),
                                          topology = (Periodic, Bounded, Bounded))
#=
grid = RegularRectilinearGrid(size = (Nx, Ny, Nz),
                              halo = (3, 3, 3), 
                              x = (0, Lx),  # longer in streamwise direction
                              y = (0, Ly),  # wide enough to avoid finite-width effects?
                              z = (-Lz, 0), # not full depth, but deep enough?
                              topology = (Periodic, Bounded, Bounded))
=#

@show grid

#####
##### Parameters
#####

a = 0.000      # m, reference wave amplitude at t ≈ τᵃ
τ = 1e-4       # m² s⁻², surface stress
k = 2π / 0.02  # m⁻¹, constant wavenumber
ν = 1.05e-6    # m² s⁻¹, kinematic viscosity
κ = 1e-6       # tracer diffusivity

struct ConstantStokesShear{T}
    a :: T
    k :: T
    ω :: T
end

ConstantStokesShear(a, k; g = 9.81, T = 7.2e-5) =
    ConstantStokesShear{Float64}(a, k, sqrt(g * k + T * k^3))

(sh::ConstantStokesShear)(z, t) = 2 * sh.a^2 * sh.k^2 * sh.ω * exp(2 * sh.k * z) 

∂z_uˢ = ConstantStokesShear(a, k)

# Calculations
ω = ∂z_uˢ.ω
ϵ = a * k
u★ = sqrt(abs(τ))
La = k * ν^(3/2) / (a * u★ * √(ω))

@info """

    Wave parameters | Values
    =============== | ======
                  a | $a
                  k | $k
             2π / k | $(2π / k)
                  ϵ | $ϵ
                 La | $La
               La⁻¹ | $(1 / La)
"""


#u_top_bc = FluxBoundaryCondition((x, y, t) -> - 1e-5 * sqrt(t))
u_top_bc = FluxBoundaryCondition(-τ)
u_bcs = FieldBoundaryConditions(top = u_top_bc)
boundary_conditions = (; u = u_bcs)

@info "Modeling..."

model = NonhydrostaticModel(architecture = arch,
                            advection = WENO5(),
                            timestepper = :RungeKutta3,
                            grid = grid,
                            tracers = :c,
                            boundary_conditions = boundary_conditions,
                            closure = IsotropicDiffusivity(ν=ν, κ=κ),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                            coriolis = nothing,
                            buoyancy = nothing)

Δz = try
    grid.Δz # for a regular grid
catch 
    minimum(parent(grid.Δzᵃᵃᶜ)) # for a stretched grid
end

uᵢ(x, y, z) = 1e-9 * randn()
set!(model, u=uᵢ, v=uᵢ, w=uᵢ)

c = model.tracers.c
view(interior(c), :, :, grid.Nz) .= 1

@info "Revvving up a simulation..."
                                    
simulation = Simulation(model, Δt=0.01, stop_time=1minutes)

wizard = TimeStepWizard(cfl=0.5, max_Δt=1.0)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

function progress(sim)

    umax = maximum(abs, sim.model.velocities.u)
    vmax = maximum(abs, sim.model.velocities.v)
    wmax = maximum(abs, sim.model.velocities.w)

    t = time(sim)
    h = √(ν * t)
    Re = umax * h / ν

    @info @sprintf("Time: %s, iteration: %d, next Δt: %s, max|U|: (%.2e, %.2e, %.2e)  m s⁻¹, hk: %.2f, Re: %.1f",
                   prettytime(t),
                   iteration(sim),
                   prettytime(sim.Δt),
                   umax, vmax, wmax,
                   h * k,
                   Re)

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

#####
##### Set up output
#####

prefix = @sprintf("simple_instability_%d_%d_%d_τ%.1e_a%.1e", grid.Nx, grid.Ny, grid.Nz, τ, a)

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

@time run!(simulation)

@info "Simulation complete: $simulation. Output:"

for (name, writer) in simulation.output_writers
    absfilepath = abspath(writer.filepath)
    @info "OutputWriter $name, $absfilepath:\n $writer"
end
