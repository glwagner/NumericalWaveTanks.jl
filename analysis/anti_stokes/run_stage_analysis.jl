#####
##### One-session analysis of a completed stage: acceptance reports for the nulls and every
##### seed, the ensemble figure, and (for the first seed) the paired figure, Hovmöller,
##### turbulence statistics, momentum budget, and the animation if slices were written.
#####
##### Usage: julia --project=. analysis/anti_stokes/run_stage_analysis.jl case=1.D level=M0 seeds=1,2,3,4
#####            [numerics=weno] [dt=0.02] [level2=M1 seeds2=1,2,3,4] [animation=true] [root=<dir>]
#####

using CairoMakie
include("quick_checks.jl")

const stage_args = parse_key_value_args(ARGS)
level = getarg(stage_args, "level", "M0")
seeds = parse.(Int, split(getarg(stage_args, "seeds", "1,2,3,4"), ','))
numerics = getarg(stage_args, "numerics", "weno")
Δt = getarg(stage_args, "dt", 0.02)
root = getarg(stage_args, "root", default_data_root())
make_animation = getarg(stage_args, "animation", true)
case_name = getarg(stage_args, "case", "1.D")
case = anti_stokes_case(case_name)
x_topology = getarg(stage_args, "x_topology", "periodic")

dir(member; seed=0, lvl=level, num=numerics, dt=Δt) = run_directory(root, case, lvl, member; seed, Δt=dt, numerics=num, x_topology)

function safely(f, description)
    try
        f()
    catch err
        @error "$description failed" exception=(err, catch_backtrace())
    end
end

# Run a figure script with the given key=value arguments in this session
function run_script(name, kv...)
    empty!(ARGS)
    append!(ARGS, collect(kv))
    safely(() -> include(joinpath(@__DIR__, name)), name)
end

#####
##### Reports
#####

null = dir("packet_null")
quiescent = dir("quiescent_control")
isdir(quiescent) || (quiescent = nothing)

isnothing(quiescent) || safely(() -> quiescent_report(quiescent), "quiescent report")
safely(() -> packet_null_report(null), "null report")
half = dir("packet_null"; dt=Δt / 2)
isdir(half) && safely(() -> null_convergence_report(null, half), "null convergence report")

for seed in seeds
    safely(() -> turbulence_report(dir("turbulence_control"; seed)), "turbulence report seed $seed")
    safely(() -> pair_report(dir("packet_turbulence"; seed), dir("turbulence_control"; seed), null, quiescent), "pair report seed $seed")
end

#####
##### Figures
#####

seed₁ = first(seeds)
pk₁, ct₁ = dir("packet_turbulence"; seed=seed₁), dir("turbulence_control"; seed=seed₁)
qkv = isnothing(quiescent) ? () : ("quiescent=$quiescent",)

ens = ["case=$case_name", "level=$level", "seeds=" * join(seeds, ','), "numerics=$numerics", "dt=$Δt", "root=$root", "x_topology=$x_topology"]
haskey(stage_args, "level2") && append!(ens, ["level2=$(stage_args["level2"])", "seeds2=$(get(stage_args, "seeds2", join(seeds, ',')))"])
run_script("ensemble_profile.jl", ens...)
run_script("compare_packet_control.jl", "packet=$pk₁", "control=$ct₁", "null=$null", qkv...)
run_script("packet_hovmoller.jl", "run=$null")
run_script("packet_hovmoller.jl", "run=$pk₁", "control=$ct₁")
run_script("turbulence_statistics.jl", "control=$ct₁", "packet=$pk₁")
run_script("momentum_budget.jl", "packet=$pk₁", "control=$ct₁", "null=$null", qkv...)

if make_animation && isfile(joinpath(pk₁, "xy_surface.jld2")) && isfile(joinpath(ct₁, "xy_surface.jld2"))
    run_script("animate_packet_turbulence.jl", "packet=$pk₁", "control=$ct₁")
end

@info "Stage $level analysis complete; figures in $(figure_directory())"
