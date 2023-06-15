using GLMakie
using JLD2
using MAT
using Oceananigans

set_theme!(Theme(fontsize=24))

include("veron_melville_data.jl")

t₀_vm = 0.0 #19.4
t_vm_surf = veron_melville_data[:t_surf] .- t₀_vm
u_vm_surf = veron_melville_data[:u_surf] ./ 100

t_vm_avg_surf = veron_melville_data[:t_avg_surf] .- t₀_vm
u_vm_avg_surf = veron_melville_data[:u_avg_surf] ./ 100

t_vm_jet = veron_melville_data[:t_jet] .- t₀_vm
u_vm_jet = veron_melville_data[:u_jet] ./ 100

t_vm_wake = veron_melville_data[:t_wake] .- t₀_vm
u_vm_wake = veron_melville_data[:u_wake] ./ 100

#####
##### Figure
#####

t₀ = 0
t₁ = 30
αvm = 0.7

fig = Figure(resolution=(1200, 1600))
ax_u = Axis(fig[1, 1], xlabel="Time relative to turbulent transition (seconds)", ylabel="Maximum surface velocity (m s⁻¹)")
xlims!(ax_u, t₀, t₁)

scatter!(ax_u, t_vm_surf, u_vm_surf, marker=:utriangle, markersize=20, color=(:blue, αvm),
         label="Surface velocity, Veron and Melville (2001)")

scatter!(ax_u, t_vm_avg_surf, u_vm_avg_surf, marker=:rect, markersize=20, color=(:purple, αvm),
         label="Average surface velocity, Veron and Melville (2001)")

scatter!(ax_u, t_vm_jet, u_vm_jet, markersize=10, color=(:indigo, αvm),
         label="Jet velocity, Veron and Melville (2001)")

scatter!(ax_u, t_vm_wake, u_vm_wake, marker=:cross, markersize=20, color=(:cyan, 1.0),
         label="Wake velocity, Veron and Melville (2001)")

#####
##### Load simulation data
#####

function compute_timeseries(filepath)
    statsfile = jldopen(filepath)
    iters = parse.(Int, keys(statsfile["timeseries/t"]))
    timeseries = Dict()
    timeseries[:filepath] = filepath
    timeseries[:grid] = statsfile["serialized/grid"]

    timeseries[:t] = t = [statsfile["timeseries/t/$i"] for i in iters]
    I = sortperm(t)
    #t = t[I]

    for stat in (:u_max, :u_min, :v_max, :w_max)
        timeseries[stat] = [statsfile["timeseries/$stat/$i"] for i in iters]
        #timeseries[stat] = timeseries[stat][I]
    end

    close(statsfile)

    return timeseries
end

dir = "data"
Δ_max = 1e-3

#cases = ["increasing_wind_ep18_k30_beta110_N512_512_384_L10_10_5_hi_freq",
#         "increasing_wind_ep18_k30_beta110_N384_384_256_L10_10_5_hi_freq",
#         "increasing_wind_ep18_k30_beta110_N256_256_192_L10_10_5_hi_freq"]

cases = [
         "increasing_wind_ep14_k30_beta110_N256_256_192_L10_10_5_hi_freq",
         "increasing_wind_ep14_k30_beta110_N384_384_256_L10_10_5_hi_freq",
        ]

colors = [:red, :orange, :black]
labels = ["256² × 192", "384² × 256", "512² × 384"]
αsim = 0.6

for n = 1:length(cases)
    case = cases[n]
    color = colors[n]
    label = labels[n]

    statistics_filename = case * "_statistics.jld2"
    averages_filename   = case * "_averages.jld2"

    statistics_filepath = joinpath(dir, statistics_filename)
    averages_filepath   = joinpath(dir, averages_filename)

    U = FieldTimeSeries(averages_filepath, "u")
    Nz = U.grid.Nz
    u_avg = [U[n][1, 1, Nz] for n in 1:length(U.times)]
    t_avg = U.times

    stats = compute_timeseries(statistics_filepath)
    t_stats = stats[:t]
    u_max = stats[:u_max]
    u_min = stats[:u_min]

    #=
    # Shift time according to turbulent transition
    nn = sortperm(t_stats)
    t_stats = t_stats[nn]
    u_max = u_max[nn]

    n_transition = findfirst(i -> u_max[i+1] < u_max[i] - Δ_max, 1:length(u_max)-1)
    t_transition = t_stats[n_transition]
    t_stats = t_stats .- t_transition
    t_avg = t_avg .- t_transition
    @show t_transition
    =#

    ct = FieldTimeSeries(averages_filepath, "c")
    c_sim = interior(ct, 1, 1, :, :)
    z_sim = znodes(Center, ct.grid)
    t_sim = ct.times .- t_transition
    Nt = length(t_sim)

    lines!(ax_u, t_stats, u_max, linewidth=6, color=(color, αsim/2))
    lines!(ax_u, t_stats, u_min, linewidth=6, color=(color, αsim/2))
    lines!(ax_u, t_avg, u_avg; linewidth=4, color=(color, αsim), label)
end

Legend(fig[0, 1], ax_u, tellwidth=false)

display(fig)

