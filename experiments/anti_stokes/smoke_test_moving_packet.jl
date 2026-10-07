#####
##### CPU plumbing test for the moving-packet experiment. Runs a tiny packet-only null
##### (level T0, Float64) for a few group widths and checks that
#####
#####   * the run completes without NaNs and writes every output file;
#####   * the sampled surface Stokes drift at the FOV follows the analytic envelope;
#####   * the quiescent control stays at rest.
#####
##### Usage: julia --project=. experiments/anti_stokes/smoke_test_moving_packet.jl [root=<dir>]
#####

include("moving_packet_experiment.jl")
include(joinpath(@__DIR__, "..", "..", "analysis", "anti_stokes", "quick_checks.jl"))

args = parse_key_value_args(ARGS)
root = getarg(args, "root", joinpath(@__DIR__, "..", "..", "data", "anti_stokes_smoke"))
stop_time = getarg(args, "stop_time", 4.0)

# The T0 level keeps the 12 m tank so that packet parameters are unchanged.
sim_q, dir_q = run_member(; member="quiescent_control", level="T0", FT=Float64, arch=CPU(),
                            root, stop_time, output_interval=0.1, progress_interval=20)

sim_p, dir_p = run_member(; member="packet_null", level="T0", FT=Float64, arch=CPU(),
                            root, stop_time, output_interval=0.1, progress_interval=20)

for dir in (dir_q, dir_p), file in ("y_averages.jld2", "fov_plane.jld2", "statistics.jld2", "snapshots.jld2",
                                    "metadata.jld2", "run_summary.jld2")
    isfile(joinpath(dir, file)) || error("Missing output $file in $dir")
end

quiescent_report(dir_q)
report = packet_null_report(dir_p)

report.trajectory_error < 1e-6 || error("Packet trajectory error $(report.trajectory_error) too large")
@info "Smoke test passed"
