#####
##### Shared configuration for the anti-Stokes moving-packet campaign:
##### resolution ladder, grid, model, numerics options, key=value command-line
##### parsing, output-directory layout, metadata, and a progress callback.
#####

using CUDA  # required for GPU(): Oceananigans 0.107 defines it in its CUDA extension
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: on_architecture
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization
using Oceananigans.Utils: AbstractSchedule
using JLD2
using Printf

# JLD2's default memory-mapped IO fails with "msync: Invalid argument" on the Lustre
# /work filesystem; use stream IO for every JLD2 file written by the campaign.
const JLD2_KW = Dict{Symbol, Any}(:iotype => IOStream)
using Statistics
using SHA
using Dates

include("cases.jl")
include("moving_packet.jl")

#####
##### Resolution ladder (see campaign document, section 7)
#####

const RESOLUTION_LEVELS = Dict(
    "T0"  => (Nx = 96,   Ny = 8,   Nz = 16),   # CPU plumbing test only
    "S0"  => (Nx = 384,  Ny = 32,  Nz = 48),   # smoke tests and packet-only null
    "S0x" => (Nx = 768,  Ny = 32,  Nz = 48),   # S0 with doubled x-resolution (null convergence)
    "M0"  => (Nx = 768,  Ny = 64,  Nz = 64),   # first turbulent signal
    "M1"  => (Nx = 1024, Ny = 96,  Nz = 96),   # first resolution comparison
    "M2"  => (Nx = 1536, Ny = 128, Nz = 128),  # quantitative profiles and budgets
    "M3"  => (Nx = 2048, Ny = 192, Nz = 160),  # selected converged cases
)

function level_size(level)
    haskey(RESOLUTION_LEVELS, level) ||
        error("Unknown level \"$level\". Known levels: $(join(sort(collect(keys(RESOLUTION_LEVELS))), ", "))")
    return RESOLUTION_LEVELS[level]
end

#####
##### Experiment members
#####

const MEMBERS = ("quiescent_control", "packet_null", "turbulence_control", "packet_turbulence")

has_packet(member) = member in ("packet_null", "packet_turbulence")
has_turbulence(member) = member in ("turbulence_control", "packet_turbulence")

function validate_member(member)
    member in MEMBERS || error("Unknown member \"$member\". Known members: $(join(MEMBERS, ", "))")
    return member
end

#####
##### Grid and model
#####

"""
    stretched_z_faces(Nz, Lz; refinement=1.5, stretching=8)

Vertical face-coordinate function used by the existing wave-tank experiments:
refined near the surface, stretched toward the bottom, spanning (-Lz, 0).
"""
function stretched_z_faces(Nz, Lz; refinement=1.5, stretching=8)
    h(k)  = (k - 1) / Nz
    ζ₀(k) = 1 + (h(k) - 1) / refinement
    Σ(k)  = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))
    return k -> Lz * (ζ₀(k) * Σ(k) - 1)
end

function build_grid(arch, FT; Nx, Ny, Nz, Lx=12, Ly=0.8, Lz=0.4, halo=3)
    z = stretched_z_faces(Nz, Lz)
    return RectilinearGrid(arch, FT;
                           size = (Nx, Ny, Nz),
                           halo = (halo, halo, halo),
                           x = (0, Lx),
                           y = (0, Ly),
                           z,
                           topology = (Periodic, Periodic, Bounded))
end

"""
    numerics_settings(name, FT=Float32)

Advection scheme, closure, and required halo for a named numerical configuration:

* `"weno"`    — WENO(order=5) implicit LES, no explicit closure (baseline)
* `"amd"`     — Centered(order=2) with AnisotropicMinimumDissipation (closure sensitivity)
* `"weno_nu"` — WENO(order=5) with molecular viscosity ν = 1.05e-6 (higher-resolution sensitivity)
* `"weno9"`   — WENO(order=9) implicit LES
"""
function numerics_settings(name, FT=Float32)
    if name == "weno"
        return (; advection = WENO(FT; order=5), closure = nothing, halo = 3)
    elseif name == "weno9"
        return (; advection = WENO(FT; order=9), closure = nothing, halo = 5)
    elseif name == "amd"
        return (; advection = Centered(FT; order=2), closure = AnisotropicMinimumDissipation(FT), halo = 3)
    elseif name == "weno_nu"
        return (; advection = WENO(FT; order=5), closure = ScalarDiffusivity(ν=FT(1.05e-6)), halo = 3)
    else
        error("Unknown numerics \"$name\". Known: weno, amd, weno_nu, weno9")
    end
end

"""
    build_model(grid; stokes_drift=nothing, advection=WENO(order=5), closure=nothing)

Clean Craik–Leibovich model: no wind stress, no buoyancy, no Coriolis, no tracers,
no Stokes-streaming forcing. The packet's time dependence lives in `stokes_drift`.
"""
function build_model(grid; stokes_drift=nothing, advection=WENO(order=5), closure=nothing)
    return NonhydrostaticModel(grid;
                               advection,
                               timestepper = :RungeKutta3,
                               closure,
                               stokes_drift,
                               buoyancy = nothing,
                               coriolis = nothing,
                               tracers = ())
end

#####
##### Schedules
#####

"""
    SpecifiedIterations(iterations)

Fires at the listed iteration numbers. Used instead of `SpecifiedTimes` so that
the fixed timestep is never perturbed by output alignment, which keeps all members
of a paired experiment on identical time grids.
"""
struct SpecifiedIterations <: AbstractSchedule
    iterations :: Vector{Int}
end

(schedule::SpecifiedIterations)(model) = model.clock.iteration ∈ schedule.iterations

Base.summary(schedule::SpecifiedIterations) = "SpecifiedIterations($(schedule.iterations))"

#####
##### key=value command-line parsing
#####

function parse_key_value_args(args=ARGS)
    parsed = Dict{String, String}()
    for arg in args
        occursin('=', arg) || error("Argument \"$arg\" is not of the form key=value")
        key, value = split(arg, '='; limit=2)
        parsed[String(key)] = String(value)
    end
    return parsed
end

convert_arg(::Type{String}, s) = s
convert_arg(::Type{Bool}, s) = parse(Bool, s)
convert_arg(::Type{T}, s) where T <: Number = parse(T, s)

getarg(args, key, default::T) where T = haskey(args, key) ? convert_arg(T, args[key]) : default

function float_type(name)
    name == "Float32" && return Float32
    name == "Float64" && return Float64
    error("Unknown float type \"$name\"")
end

function architecture(name)
    name == "gpu" && return GPU()
    name == "cpu" && return CPU()
    error("Unknown architecture \"$name\" (use gpu or cpu)")
end

#####
##### Directory layout
#####
#####   root/case_1D/M0/initial_conditions/seed_0001.jld2
#####   root/case_1D/M0/packet_null/quiescent_dt0.020/
#####   root/case_1D/M0/turbulence_control/seed_0001_dt0.020/
#####   root/case_1D/M0/packet_turbulence/seed_0001_dt0.020/
#####

function default_data_root()
    work = "/work/hdd/bhcr/glwagner"
    isdir(work) && return joinpath(work, "anti_stokes")
    return normpath(joinpath(@__DIR__, "..", "..", "data", "anti_stokes"))
end

case_dirname(case) = "case_" * replace(case.name, "." => "")
level_dir(root, case, level) = joinpath(root, case_dirname(case), level)

domain_tag(Lx, Ly) = (Lx == 12 && Ly == 0.8) ? "" : @sprintf("Lx%g_Ly%g", Lx, Ly)

function initial_condition_path(root, case, level, seed; Lx=12, Ly=0.8)
    parts = filter(!isempty, [@sprintf("seed_%04d", seed), domain_tag(Lx, Ly)])
    return joinpath(level_dir(root, case, level), "initial_conditions", join(parts, "_") * ".jld2")
end

function run_directory(root, case, level, member; seed, Δt, numerics, Lx=12, Ly=0.8, extra="")
    parts = [has_turbulence(member) ? @sprintf("seed_%04d", seed) : "quiescent"]
    numerics == "weno" || push!(parts, numerics)
    push!(parts, @sprintf("dt%.3f", Δt))
    tag = domain_tag(Lx, Ly)
    isempty(tag) || push!(parts, tag)
    isempty(extra) || push!(parts, extra)
    return joinpath(level_dir(root, case, level), member, join(parts, "_"))
end

#####
##### Metadata
#####

git_commit() = try readchomp(`git -C $(@__DIR__) rev-parse HEAD`) catch; "unknown" end
git_dirty() = try !isempty(readchomp(`git -C $(@__DIR__) status --porcelain`)) catch; true end
oceananigans_version() = string(pkgversion(Oceananigans))
file_sha256(path) = open(io -> bytes2hex(sha256(io)), path)

#####
##### Progress callback and diagnostics
#####

function make_progress(packet=nothing)
    wall = Ref(time_ns())
    function progress(sim)
        u, v, w = sim.model.velocities
        umax = maximum(abs, u)
        vmax = maximum(abs, v)
        wmax = maximum(abs, w)
        t = time(sim)
        elapsed = 1e-9 * (time_ns() - wall[])
        msg = @sprintf("iter: %6d, t: %8.3f s, wall: %s, max|u,v,w| = (%.3e, %.3e, %.3e)",
                       iteration(sim), t, prettytime(elapsed), umax, vmax, wmax)
        isnothing(packet) || (msg *= @sprintf(", x_c = %.3f m", packet_center(t, packet)))
        @info msg
        wall[] = time_ns()
        isfinite(umax) && isfinite(vmax) && isfinite(wmax) ||
            error("Non-finite velocity at iteration $(iteration(sim))")
        return nothing
    end
    return progress
end

"""
    volume_mean(field)

Volume-weighted mean of a field (the plain `mean` over cells over-weights the refined
near-surface cells of the stretched grid).
"""
volume_mean(field) = first(Array(interior(compute!(Field(Average(field))))))

function component_rms(model)
    u, v, w = model.velocities
    # sum/length rather than Statistics.mean: mean(f, A) scalar-indexes GPU arrays via first(A)
    rms(f) = sqrt(sum(abs2, interior(f)) / length(interior(f)))
    return (u = rms(u), v = rms(v), w = rms(w))
end
