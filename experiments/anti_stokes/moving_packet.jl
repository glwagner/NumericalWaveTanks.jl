#####
##### Prescribed, solenoidal, long-crested Gaussian Stokes-drift packet moving
##### at the group velocity through a periodic tank.
#####
##### The wave-amplitude envelope is a(ξ) = aₚ exp(-ξ² / 2σ₀²), so the Stokes
##### envelope (∝ a²) is G(ξ) = exp(-ξ² / σ₀²), with ξ = x - x_c(t) and
##### x_c(t) = x₀ + cᵍ t. The horizontal Stokes drift is
#####
#####     uˢ = Uˢ₀ G(ξ) exp(2kz),
#####
##### and the finite-depth solenoidal completion with wˢ = 0 at z = -h is
#####
#####     wˢ = -Uˢ₀ / (2k) G′(ξ) [exp(2kz) - exp(-2kh)].
#####
##### G is periodized with the three nearest images so the envelope is smooth
##### when the packet straddles the streamwise boundary.
#####

using Oceananigans

@inline gaussian(d, σ)   = exp(-abs2(d / σ))
@inline gaussian′(d, σ)  = -2d / σ^2 * gaussian(d, σ)
@inline gaussian′′(d, σ) = (4d^2 / σ^4 - 2 / σ^2) * gaussian(d, σ)

# Three nearest periodic images. For Lx / σ₀ ≈ 8.35 the omitted images
# are far below Float32 precision anywhere in the domain.
@inline G(ξ, p)   = gaussian(ξ - p.Lx, p.σ₀)   + gaussian(ξ, p.σ₀)   + gaussian(ξ + p.Lx, p.σ₀)
@inline G′(ξ, p)  = gaussian′(ξ - p.Lx, p.σ₀)  + gaussian′(ξ, p.σ₀)  + gaussian′(ξ + p.Lx, p.σ₀)
@inline G′′(ξ, p) = gaussian′′(ξ - p.Lx, p.σ₀) + gaussian′′(ξ, p.σ₀) + gaussian′′(ξ + p.Lx, p.σ₀)

@inline packet_center(t, p) = p.x₀ + p.cᵍ * t
@inline packet_coordinate(x, t, p) = x - packet_center(t, p)

@inline E(z, p) = exp(2p.k * z)
@inline F(z, p) = E(z, p) - exp(-2p.k * p.h)

# Values, used for diagnostics and initialization.
@inline function uˢ(x, y, z, t, p)
    ξ = packet_coordinate(x, t, p)
    return p.Uˢ₀ * G(ξ, p) * E(z, p)
end

@inline function wˢ(x, y, z, t, p)
    ξ = packet_coordinate(x, t, p)
    return -p.Uˢ₀ / (2p.k) * G′(ξ, p) * F(z, p)
end

# Derivatives passed to Oceananigans.StokesDrift.
@inline function ∂z_uˢ(x, y, z, t, p)
    ξ = packet_coordinate(x, t, p)
    return 2p.k * p.Uˢ₀ * G(ξ, p) * E(z, p)
end

@inline function ∂t_uˢ(x, y, z, t, p)
    ξ = packet_coordinate(x, t, p)
    return -p.cᵍ * p.Uˢ₀ * G′(ξ, p) * E(z, p)
end

@inline function ∂x_wˢ(x, y, z, t, p)
    ξ = packet_coordinate(x, t, p)
    return -p.Uˢ₀ / (2p.k) * G′′(ξ, p) * F(z, p)
end

@inline function ∂t_wˢ(x, y, z, t, p)
    ξ = packet_coordinate(x, t, p)
    return p.cᵍ * p.Uˢ₀ / (2p.k) * G′′(ξ, p) * F(z, p)
end

# Additional derivatives used only by tests and diagnostics
@inline function ∂x_uˢ(x, y, z, t, p)
    ξ = packet_coordinate(x, t, p)
    return p.Uˢ₀ * G′(ξ, p) * E(z, p)
end

@inline function ∂z_wˢ(x, y, z, t, p)
    ξ = packet_coordinate(x, t, p)
    return -p.Uˢ₀ * G′(ξ, p) * E(z, p)
end

"""
    packet_parameters(case, Lx, x_FOV; σ_upstream=4)

Packet parameters for a case: the packet center starts `σ_upstream` widths upstream
of the observation plane `x_FOV` in a periodic tank of length `Lx`.
"""
function packet_parameters(case, Lx, x_FOV; σ_upstream=4)
    FT = typeof(case.k)
    x₀ = FT(x_FOV - σ_upstream * case.σ₀)
    return (; k = case.k, h = case.h, cᵍ = case.cᵍ, Uˢ₀ = case.Uˢ₀, σ₀ = case.σ₀,
              x₀, Lx = FT(Lx), x_FOV = FT(x_FOV))
end

"""
    moving_stokes_packet(case, Lx, x_FOV; kw...)

Return `(stokes_drift, parameters)` for a long-crested Gaussian Stokes packet
moving at the group velocity `case.cᵍ`.
"""
function moving_stokes_packet(case, Lx, x_FOV; kw...)
    parameters = packet_parameters(case, Lx, x_FOV; kw...)
    stokes_drift = StokesDrift(; ∂z_uˢ, ∂t_uˢ, ∂x_wˢ, ∂t_wˢ, parameters)
    return stokes_drift, parameters
end

# Time at which the packet peak crosses x_FOV, and the time at which it has
# moved the same distance downstream.
packet_peak_time(p) = (p.x_FOV - p.x₀) / p.cᵍ
packet_stop_time(p) = 2 * packet_peak_time(p)

"""
    stokes_drift_fields(grid, p, t=0)

Evaluate `uˢ` and `wˢ` at time `t` on the native staggered nodes of `grid`,
returning `(uˢ_field, wˢ_field)`.
"""
function stokes_drift_fields(grid, p, t=0)
    u_field = XFaceField(grid)
    w_field = ZFaceField(grid)
    set!(u_field, (x, y, z) -> uˢ(x, y, z, t, p))
    set!(w_field, (x, y, z) -> wˢ(x, y, z, t, p))
    return u_field, w_field
end

"""
    stokes_drift_xz(x, z, t, p)

Arrays `uˢ(x, z)` and `wˢ(x, z)` for vectors of coordinates, used for offline
Eulerian subtraction from y-averaged output.
"""
function stokes_drift_xz(x, z, t, p)
    U = [uˢ(xi, 0, zk, t, p) for xi in x, zk in z]
    W = [wˢ(xi, 0, zk, t, p) for xi in x, zk in z]
    return U, W
end
