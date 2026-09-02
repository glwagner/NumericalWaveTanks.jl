#####
##### Command-line entry point for one experiment member, e.g.
#####
#####   julia --project=. experiments/anti_stokes/run_moving_packet.jl case=1.D member=packet_null level=S0 seed=0
#####   julia --project=. experiments/anti_stokes/run_moving_packet.jl case=1.D member=turbulence_control level=M0 seed=1
#####   julia --project=. experiments/anti_stokes/run_moving_packet.jl case=1.D member=packet_turbulence level=M0 seed=1
#####
##### Optional keys: dt=0.02 numerics=weno|amd|weno_nu|weno9 arch=gpu|cpu FT=Float32|Float64
#####                Lx=12 Ly=0.8 x_fov=6 stop_time=22.4 output_interval=0.1 root=<dir>
#####                overwrite=true tag=<extra directory suffix> animation=false (x-z and surface slices)
#####

include("moving_packet_experiment.jl")

args = parse_key_value_args(ARGS)

Lx = getarg(args, "Lx", 12.0)
stop_time = haskey(args, "stop_time") ? getarg(args, "stop_time", 0.0) : nothing

run_member(; case_name = getarg(args, "case", "1.D"),
             member = getarg(args, "member", "packet_null"),
             level = getarg(args, "level", "S0"),
             seed = getarg(args, "seed", 0),
             Δt = getarg(args, "dt", 0.02),
             numerics = getarg(args, "numerics", "weno"),
             FT = float_type(getarg(args, "FT", "Float32")),
             arch = architecture(getarg(args, "arch", "gpu")),
             Lx,
             Ly = getarg(args, "Ly", 0.8),
             x_FOV = getarg(args, "x_fov", Lx / 2),
             σ_upstream = getarg(args, "sigma_upstream", 4),
             stop_time,
             output_interval = getarg(args, "output_interval", 0.1),
             animation_slices = getarg(args, "animation", false),
             root = getarg(args, "root", default_data_root()),
             overwrite = getarg(args, "overwrite", true),
             progress_interval = getarg(args, "progress_interval", 50),
             tag_extra = getarg(args, "tag", ""))
