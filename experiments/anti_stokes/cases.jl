#####
##### Experimental cases from Ellingsen et al. (2026), "Turbulence-induced
##### anti-Stokes flow: experiments and theory", JFM 1029, A6.
#####
##### Only directly reported quantities are stored; everything else is derived
##### in one place so that no rounded intermediate values are scattered around.
#####

"""
    case_1D(FT=Float32)

Experiment 1, case 1.D: the wave-group case with the strongest turbulence-induced
anti-Stokes signal. Returns a NamedTuple containing the measured quantities and
derived packet parameters (phase speed `c`, group speed `cᵍ`, surface Stokes drift
`Uˢ₀`, intrinsic spatial half-width `σ₀`, Stokes e-folding depth `δˢ`, and effective
interaction time `Tᵢ`).
"""
function case_1D(FT=Float32)
    g = FT(9.81)

    measured = (; name      = "1.D",
                  h         = FT(0.40),   # water depth [m]
                  U₀        = FT(0.34),   # mean current magnitude [m s⁻¹]
                  k         = FT(9.3),    # carrier wavenumber [m⁻¹]
                  τ_lab     = FT(7.8),    # laboratory group duration [s]
                  τ₀        = FT(2.8),    # intrinsic group duration [s]
                  u_rms     = FT(0.017),  # streamwise rms velocity [m s⁻¹]
                  v_rms     = FT(0.017),  # spanwise rms velocity [m s⁻¹]
                  w_rms     = FT(0.013),  # vertical rms velocity [m s⁻¹]
                  e         = FT(3.7e-4), # turbulent kinetic energy [m² s⁻²]
                  L         = FT(0.20),   # streamwise integral scale Lₓˣ [m]
                  steepness = FT(0.22))   # peak steepness ϵ = k₀ aₚ

    c   = sqrt(g / measured.k)            # deep-water phase speed
    cᵍ  = c / 2                           # deep-water group speed
    Uˢ₀ = measured.steepness^2 * c        # surface Stokes drift at packet peak
    σ₀  = cᵍ * measured.τ₀                # intrinsic spatial amplitude width
    δˢ  = 1 / (2 * measured.k)            # Stokes e-folding depth
    Tᵢ  = sqrt(FT(π)) * measured.τ₀       # effective interaction time

    return merge(measured, (; g, c, cᵍ, Uˢ₀, σ₀, δˢ, Tᵢ))
end

const ANTI_STOKES_CASES = Dict("1.D" => case_1D, "1D" => case_1D)

"""
    anti_stokes_case(name, FT=Float32)

Look up a case by name (for example `"1.D"`). Cases 1.A–1.C, 2 and 3 are part of the
later physics campaign and are added here once their tabulated values are transcribed.
"""
function anti_stokes_case(name, FT=Float32)
    haskey(ANTI_STOKES_CASES, name) ||
        error("Unknown case \"$name\". Known cases: $(join(sort(collect(keys(ANTI_STOKES_CASES))), ", "))")
    return ANTI_STOKES_CASES[name](FT)
end
