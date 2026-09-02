##### Stage S0 (campaign document, section 12): implementation and null physics.
##### Runs every S0 member in one Julia session (one GPU) to amortize compilation:
#####
#####   1. quiescent control                      5. one crude turbulent control (seed 1)
#####   2. packet-only null                       6. one crude packet+turbulence member (seed 1)
#####   3. packet-only null, half timestep
#####   4. packet-only null, doubled x-resolution
#####
##### then prints the Makie-free acceptance report from analysis/anti_stokes/quick_checks.jl.
##### No scientific conclusion should be drawn from S0.
#####

include("moving_packet_experiment.jl")

args = parse_key_value_args(ARGS)
root = getarg(args, "root", default_data_root())
arch = architecture(getarg(args, "arch", "gpu"))
level = getarg(args, "level", "S0")
seed = getarg(args, "seed", 1)
amplitude = parse_amplitude(getarg(args, "amplitude", "1.6"))
skip = split(getarg(args, "skip", ""), ',')  # comma-separated list of step numbers to skip

timings = Dict{String, Float64}()
dirs = Dict{String, String}()

function step!(name, number, f)
    string(number) in skip && (@info "Skipping step $number ($name)"; return)
    @info "===== S0 step $number: $name ====="
    wall = time_ns()
    result = f()
    timings[name] = 1e-9 * (time_ns() - wall)
    result isa Tuple && (dirs[name] = result[2])
    @info @sprintf("===== finished %s in %s =====", name, prettytime(timings[name]))
end

step!("quiescent_control", 1, () -> run_member(; member="quiescent_control", level, arch, root))
step!("packet_null", 2, () -> run_member(; member="packet_null", level, arch, root))
step!("packet_null_dt0.010", 3, () -> run_member(; member="packet_null", level, arch, root, Δt=0.01))
step!("packet_null_S0x", 4, () -> run_member(; member="packet_null", level=level * "x", arch, root))

step!("initial_condition", 5,
      () -> generate_initial_condition(; level, seed, arch, root, amplitude, overwrite=true))

step!("turbulence_control", 6, () -> run_member(; member="turbulence_control", level, seed, arch, root))
step!("packet_turbulence", 7, () -> run_member(; member="packet_turbulence", level, seed, arch, root))

@info "S0 wall-clock summary:"
for (name, t) in sort(collect(timings); by=last)
    @info @sprintf("  %-24s %s", name, prettytime(t))
end

#####
##### Acceptance report
#####

include(joinpath(@__DIR__, "..", "..", "analysis", "anti_stokes", "quick_checks.jl"))

case = anti_stokes_case("1.D")
dir(member; kw...) = run_directory(root, case, level, member; seed, Δt=0.02, numerics="weno", kw...)

report(f, args...) = try
    f(args...)
catch err
    @error "Report $(nameof(f)) failed" exception=(err, catch_backtrace())
end

report(quiescent_report, dir("quiescent_control"))
report(packet_null_report, dir("packet_null"))
report(packet_null_report, run_directory(root, case, level, "packet_null"; seed, Δt=0.01, numerics="weno"))
report(packet_null_report, run_directory(root, case, level * "x", "packet_null"; seed, Δt=0.02, numerics="weno"))
report(null_convergence_report, dir("packet_null"),
       run_directory(root, case, level, "packet_null"; seed, Δt=0.01, numerics="weno"))
report(turbulence_report, dir("turbulence_control"))
report(pair_report, dir("packet_turbulence"), dir("turbulence_control"), dir("packet_null"), dir("quiescent_control"))
