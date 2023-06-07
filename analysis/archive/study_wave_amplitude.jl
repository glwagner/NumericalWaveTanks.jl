using GLMakie
using JLD2
using MAT
using Oceananigans
using Printf

function compute_timeseries(filepath)
    statsfile = jldopen(filepath)
    iters = parse.(Int, keys(statsfile["timeseries/t"]))
    timeseries = Dict()
    timeseries[:filepath] = filepath
    timeseries[:grid] = statsfile["serialized/grid"]

    timeseries[:t] = t = [statsfile["timeseries/t/$i"] for i in iters]

    for stat in (:u_max, :u_min, :v_max, :w_max)
        timeseries[stat] = [statsfile["timeseries/$stat/$i"] for i in iters]
    end

    close(statsfile)

    return timeseries
end

set_theme!(Theme(fontsize=24))
include("veron_melville_data.jl")

#epsilons = [14, 15, 17, 18, 19]
epsilons = [14, 17, 19]
get_filename(ep) = "increasing_wind_ep$(ep)_k30_beta110_N384_384_256_L10_10_5_hi_freq_statistics.jld2"
dir = "data"
Δ_max = 1e-3
αsim = 0.6

fig = Figure(resolution=(1800, 1600))
ax_u = Axis(fig[1, 1], ylabel="Streamwise velocity (m s⁻¹)", xaxisposition=:top, xlabel="Time relative to turbulent transition (seconds)")
ax_w = Axis(fig[2, 1], xlabel="Time relative to turbulent transition (seconds)", ylabel="Cross-stream velocities (m s⁻¹)")

for ep in epsilons
    statistics_filepath = joinpath(dir, get_filename(ep))
    stats = compute_timeseries(statistics_filepath)
    t_stats = stats[:t]

    u_max = stats[:u_max]
    v_max = stats[:v_max]
    w_max = stats[:w_max]
    u_min = stats[:u_min]

    lines!(ax_u, t_stats, u_max, linewidth=6)
    lines!(ax_w, t_stats, w_max, linewidth=6)

    # lines!(ax_u, t_stats, u_max, linewidth=6, color=(:red,    αsim), label="Simulated max(u)")
    # lines!(ax_u, t_stats, u_min, linewidth=6, color=(:orange, αsim), label="Simulated min(u)")
    # lines!(ax_w, t_stats, v_max, linewidth=6, color=(:purple, αsim), label="max(v)")

    # Legend(fig[0, 1], ax_u, tellwidth=false)
    # axislegend(ax_w, position=:rb)
    # hidespines!(ax_u, :b, :r)
    # hidespines!(ax_w, :t, :r)

    display(fig)

    save("/Users/gregorywagner/Desktop/wave_amplitude.png", fig)
end

