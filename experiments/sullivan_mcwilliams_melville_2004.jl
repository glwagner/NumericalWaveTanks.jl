# # Oceanic boundary layer driven by wave breaking
#
# This script implements a simulation inspired by Sullivan, McWilliams, and Melville (2004),
# "The oceanic boundary layer driven by wave breaking with stochastic variability",
# Journal of Fluid Mechanics, vol. 507, pp. 143-174.
#
# This script implements case 3: constant stress with Stokes drift AND stochastic breakers.

using Oceananigans
using Oceananigans.Architectures: device, architecture, on_architecture
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.Utils: launch!
using Oceananigans.Units
using KernelAbstractions: @kernel, @index
using Adapt
using Printf
using Random

#####
##### Physical parameters from Sullivan et al. (2004) / NCARLES
#####

U₁₀ = 15.0
ρₐ  = 1.0
ρw  = 1000.0
g   = 9.81
coriolis = FPlane(f = 1e-4)

Cd = (0.79 + 0.0509 * U₁₀) * 1e-3
τ_wind = ρₐ * Cd * U₁₀^2
u_star = sqrt(τ_wind / ρw)
τx = -u_star^2

Qᵇ = 4.96e-7
N² = 5e-4
h₀ = 33.0

#####
##### Stokes drift
#####

@inline ∂z_uˢ(z, t, p) = p.Uˢ / p.δ * exp(z / p.δ)

stokes_drift_parameters = (Uˢ = 0.24, δ = 6.1)
stokes_drift = UniformStokesDrift(∂z_uˢ = ∂z_uˢ, parameters = stokes_drift_parameters)

#####
##### Grid
#####

arch = CPU()  # Change to GPU() for GPU

Nx, Ny, Nz = 64, 64, 64
Lx, Ly, Lz = 300.0, 300.0, 110.0

refinement = 1.5
stretching = 4

h(k) = (k - 1) / Nz
ζ(k) = 1 + (h(k) - 1) / refinement
σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))
z_faces(k) = Lz * (ζ(k) * σ(k) - 1)

grid = RectilinearGrid(arch,
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_faces,
                       topology = (Periodic, Periodic, Bounded))

#####
##### Stochastic wave breaking model (GPU-compatible)
#####

const max_breakers = 256

# Immutable struct for GPU compatibility (descriptive property names)
struct BreakerMaker{I, A}
    active_indices::I
    x_position::A           # initial x position at spawn time
    y_position::A           # initial y position at spawn time
    phase_speed::A          # phase speed c
    propagation_angle::A    # angle φ (radians, 0 = +x direction)
    amplitude::A            # forcing amplitude coefficient (con_b)
    horizontal_scale::A     # wavelength λ
    start_time::A
    end_time::A
end

function BreakerMaker(arch)
    return BreakerMaker(
        on_architecture(arch, zeros(Int32, max_breakers)),
        on_architecture(arch, zeros(max_breakers)),
        on_architecture(arch, zeros(max_breakers)),
        on_architecture(arch, zeros(max_breakers)),
        on_architecture(arch, zeros(max_breakers)),
        on_architecture(arch, zeros(max_breakers)),
        on_architecture(arch, zeros(max_breakers)),
        on_architecture(arch, zeros(max_breakers)),
        on_architecture(arch, zeros(max_breakers)))
end

# Adapt method for GPU transfer
Adapt.adapt_structure(to, b::BreakerMaker) = BreakerMaker(
    Adapt.adapt(to, b.active_indices),
    Adapt.adapt(to, b.x_position),
    Adapt.adapt(to, b.y_position),
    Adapt.adapt(to, b.phase_speed),
    Adapt.adapt(to, b.propagation_angle),
    Adapt.adapt(to, b.amplitude),
    Adapt.adapt(to, b.horizontal_scale),
    Adapt.adapt(to, b.start_time),
    Adapt.adapt(to, b.end_time))

# CPU version for bookkeeping, GPU version for kernel
cpu_breakers = BreakerMaker(CPU())
gpu_breakers = BreakerMaker(arch)
Nb = Ref(0)  # Number of active breakers (mutable counter)

function sync_to_device!(gpu::BreakerMaker, cpu::BreakerMaker)
    copyto!(gpu.active_indices,    cpu.active_indices)
    copyto!(gpu.x_position,        cpu.x_position)
    copyto!(gpu.y_position,        cpu.y_position)
    copyto!(gpu.phase_speed,       cpu.phase_speed)
    copyto!(gpu.propagation_angle, cpu.propagation_angle)
    copyto!(gpu.amplitude,         cpu.amplitude)
    copyto!(gpu.horizontal_scale,  cpu.horizontal_scale)
    copyto!(gpu.start_time,        cpu.start_time)
    copyto!(gpu.end_time,          cpu.end_time)
end

# Generation parameters from NCARLES
c_min_les = 1.53        # minimum resolvable phase speed (based on grid)
efac = -2.047           # PDF exponential factor (case 3 in NCARLES)
c_scale_pdf = sqrt(Cd) * U₁₀  # characteristic speed for PDF
max_angle = 30.0        # Maximum propagation angle (degrees)

# Breaker amplitude: from Fortran, con_mag = con_b * T * g / (2π * dx * dy * Δt)
# We compute amplitude_base such that the forcing magnitude is O(τ/ρ/h) ~ 10^-4 m/s²
# From Sullivan et al., the amplitude is set to match the momentum flux
dx = Lx / Nx
dy = Ly / Ny
con_b_raw = 0.18
# Approximate normalization: force ~ con_b * T * g / (dx * dy)
# For c~2 m/s: T ~ 8s, so con_b * T * g / (dx * dy) ~ 0.18 * 8 * 10 / (5 * 5) ~ 0.6
# We want force ~ 10^-4, so scale down by ~6000
# Actually, let's be more careful: use dimensional analysis
# Momentum flux from breakers ~ fraction * τ_wind 
# Distributed over depth ~ λ * cz_1 ~ 3m and duration T ~ 8s
# Force ~ fraction * τ_wind / (ρw * depth * T) 
# ~ 0.5 * 0.35 / (1000 * 3 * 8) ~ 7e-6 m/s² per unit shape
amplitude_scale = τ_wind / (ρw * 10.0)  # ~ 3.5e-5 m/s², accounts for depth

# Breaker generation rate (approximate - should be computed from integral)
# This is τ / (ρw * ⟨momentum per breaker⟩)
breaker_rate = τ_wind / (ρw * g * c_scale_pdf * 50.0) * Lx * Ly

# Sample phase speed from exponential PDF: p(c) ∝ exp(efac * c / c_scale_pdf)
# Using inverse CDF sampling for c ≥ c_min_les
function sample_phase_speed()
    # Simplified: use exponential distribution with mean = -c_scale_pdf / efac
    # This gives p(c) ∝ exp(efac * c / c_scale_pdf)
    # Inverse CDF: c = c_min_les + (-c_scale_pdf / efac) * (-log(1 - u))
    #            = c_min_les - (c_scale_pdf / efac) * log(1 - u)
    u = rand()
    # efac is negative, so -c_scale_pdf/efac is positive
    scale = -c_scale_pdf / efac
    return c_min_les + scale * (-log(1 - u + 1e-10))  # add small number to avoid log(0)
end

function spawn_breaker!(b, Nb, t)
    Nb[] >= max_breakers && return
    
    c = sample_phase_speed()
    λ = 2π * g / c^2         # wavelength from deep-water dispersion
    T = λ / c                 # wave period = breaker lifetime
    
    # Random propagation angle within ±max_angle of wind direction (x)
    φ = (2 * rand() - 1) * max_angle * π / 180
    
    Nb[] += 1
    n = Nb[]
    b.active_indices[n] = Int32(n)
    b.x_position[n] = rand() * Lx
    b.y_position[n] = rand() * Ly
    b.phase_speed[n] = c
    b.propagation_angle[n] = φ
    b.amplitude[n] = amplitude_scale  # properly scaled forcing amplitude
    b.horizontal_scale[n] = λ         # store wavelength λ
    b.start_time[n] = t
    b.end_time[n] = t + T
end

function expire_breakers!(b, Nb, t)
    j = 0
    for k in 1:Nb[]
        i = b.active_indices[k]
        if t < b.end_time[i]
            j += 1
            b.active_indices[j] = i
        end
    end
    Nb[] = j
end

# Forcing fields for u and v
u_breaker_forcing = CenterField(grid)
v_breaker_forcing = CenterField(grid)

# Shape function constants from NCARLES (subroutine shapes)
const c_rise = 5.0      # temporal shape
const c_norm = 2.08206  # temporal normalization
const cx_1 = 5.0972943  # x-shape coefficient
const cx_2 = 10.0       # x-shape coefficient  
const cy_1 = 2.0        # y-shape coefficient
const cz_1 = 0.2        # z-shape depth scale (z penetration = cz_1 * λ)
const cz_2 = 2.0        # z-shape coefficient

# Temporal shape function: t_shape(α) where α ∈ [0,1]
@inline function temporal_shape(α)
    α² = α * α
    return c_norm * α² * (exp(c_rise * (1 - α)^2) - 1)
end

# X-shape (along propagation direction): x_shape(β) where β ∈ [0,1]  
@inline function x_shape(β)
    β = clamp(β, 0, 1)
    β² = β * β
    return cx_1 * (1 + cx_2 * β * β²) * β² * (1 - β)^2
end

# Y-shape (perpendicular to propagation): y_shape(δ) where δ ∈ [-1,1]
@inline function y_shape(δ)
    δ = clamp(δ, -1, 1)
    δ² = δ * δ
    return (1 - δ²)^2 * (1 + cy_1 * δ²)
end

# Z-shape (vertical): z_shape(γ) where γ ∈ [-1,0]
@inline function z_shape_func(γ)
    γ = clamp(γ, -1, 0)
    γ² = γ * γ
    return (1 - γ²)^2 * (1 + cz_2 * γ²)
end

# Kernel to compute breaker forcing using Sullivan et al. (2004) shape functions
# The breaker is a λ×λ square region that sweeps forward at phase speed c
@kernel function _compute_breaker_forcing!(Fu, Fv, grid, b, Nb, t, Lx, Ly)
    i, j, k = @index(Global, NTuple)
    
    x = xnode(i, j, k, grid, Center(), Center(), Center())
    y = ynode(i, j, k, grid, Center(), Center(), Center())
    z = znode(i, j, k, grid, Center(), Center(), Center())
    
    # Unpack to mathematical notation
    active = b.active_indices
    x₀ = b.x_position      # initial position (tail of breaker)
    y₀ = b.y_position
    c = b.phase_speed
    φ = b.propagation_angle
    A = b.amplitude        # con_b coefficient
    λ = b.horizontal_scale # wavelength (not σ!)
    t₀ = b.start_time
    t₁ = b.end_time
    
    Fᵤ = zero(eltype(Fu))
    Fᵥ = zero(eltype(Fv))
    
    @inbounds for m in 1:Nb
        idx = active[m]
        
        # Normalized time α ∈ [0,1]
        T = t₁[idx] - t₀[idx]
        α = (t - t₀[idx]) / T
        
        # Skip if outside active time or too early (avoid division by ~0)
        (α < 0.01 || α > 1) && continue
        
        # Direction cosines
        cosφ = cos(φ[idx])
        sinφ = sin(φ[idx])
        
        # Periodic distance from breaker origin
        dx = x - x₀[idx]
        dx = dx - Lx * round(dx / Lx)
        dy = y - y₀[idx]
        dy = dy - Ly * round(dy / Ly)
        
        # Transform to breaker-aligned coordinates
        # ξ = distance along propagation direction
        # η = distance perpendicular to propagation
        ξ = dx * cosφ + dy * sinφ
        η = -dx * sinφ + dy * cosφ
        
        # Normalized coordinates for shape functions
        αλ = α * λ[idx]
        β = ξ / αλ                           # along-propagation, β ∈ [0,1]
        δ = 2 * η / λ[idx]                   # cross-propagation, δ ∈ [-1,1] when |η| < λ/2
        
        # Vertical: γ = z / (α * λ * cz_1), γ ∈ [-1, 0]
        z_depth = αλ * cz_1
        γ = z / z_depth
        
        # Evaluate shape functions (they return 0 outside valid range)
        t_fun = temporal_shape(α)
        x_fun = x_shape(β)
        y_fun = y_shape(δ)
        z_fun = z_shape_func(γ)
        
        # Total forcing magnitude
        F = A[idx] * t_fun * x_fun * y_fun * z_fun
        
        # Decompose into u and v components
        Fᵤ += F * cosφ
        Fᵥ += F * sinφ
    end
    
    @inbounds Fu[i, j, k] = Fᵤ
    @inbounds Fv[i, j, k] = Fᵥ
end

function update_breaker_forcing!(sim)
    t = time(sim)
    Δt = sim.Δt
    
    # Skip if Δt is not yet set (during initialization)
    isnan(Δt) && return
    
    expire_breakers!(cpu_breakers, Nb, t)
    
    # Poisson process: expected number of new breakers
    expected_new = breaker_rate * Δt
    n_new = floor(Int, expected_new) + (rand() < (expected_new - floor(expected_new)) ? 1 : 0)
    for _ in 1:n_new
        spawn_breaker!(cpu_breakers, Nb, t)
    end
    
    sync_to_device!(gpu_breakers, cpu_breakers)
    
    launch!(arch, grid, :xyz, _compute_breaker_forcing!,
            u_breaker_forcing, v_breaker_forcing, grid, gpu_breakers, Nb[], t, Lx, Ly)
end

#####
##### Boundary conditions
#####

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ),
                                bottom = GradientBoundaryCondition(N²))

#####
##### Model
#####

model = NonhydrostaticModel(grid;
                            coriolis,
                            advection = WENO(order=5),
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            stokes_drift,
                            forcing = (; u = Forcing(u_breaker_forcing),
                                         v = Forcing(v_breaker_forcing)),
                            boundary_conditions = (u=u_bcs, b=b_bcs))

#####
##### Initial conditions
#####

Random.seed!(123)
uᵢ(x, y, z) = u_star * 1e-2 * randn() * exp(z / 10)
bᵢ(x, y, z) = z > -h₀ ? N² * (-h₀) : N² * z

set!(model, u=uᵢ, v=uᵢ, w=uᵢ, b=bᵢ)

#####
##### Simulation
#####

simulation = Simulation(model; Δt=10.0, stop_time=6hours)
conjure_time_step_wizard!(simulation, cfl=0.5, max_Δt=30.0)

simulation.callbacks[:breakers] = Callback(update_breaker_forcing!, IterationInterval(1))

function progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("i: %05d, t: %s, Δt: %s, Nb: %d, max|u|: (%.2e, %.2e, %.2e)",
                   iteration(sim), prettytime(time(sim)), prettytime(sim.Δt),
                   Nb[], maximum(abs, u), maximum(abs, v), maximum(abs, w))
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

#####
##### Output
#####

save_interval = 10minutes

u, v, w = model.velocities
b = model.tracers.b

U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
B = Average(b, dims=(1, 2))
wu = Average(w * u, dims=(1, 2))
wv = Average(w * v, dims=(1, 2))
wb = Average(w * b, dims=(1, 2))

simulation.output_writers[:averages] =
    JLD2Writer(model, (; U, V, B, wu, wv, wb);
               schedule = AveragedTimeInterval(save_interval, window=2minutes),
               filename = "sullivan2004_averages.jld2",
               overwrite_existing = true)

outputs = merge(model.velocities, model.tracers)

simulation.output_writers[:xy_surface] =
    JLD2Writer(model, outputs;
               schedule = TimeInterval(save_interval),
               filename = "sullivan2004_xy_surface.jld2",
               indices = (:, :, Nz),
               overwrite_existing = true)

simulation.output_writers[:xz_slice] =
    JLD2Writer(model, outputs;
               schedule = TimeInterval(save_interval),
               filename = "sullivan2004_xz_slice.jld2",
               indices = (:, Ny÷2, :),
               overwrite_existing = true)

#####
##### Run
#####

run!(simulation)

@info "Simulation complete!"
