using GLMakie
using JLD2
using MAT
using Oceananigans

set_theme!(Theme(fontsize=24))

include("veron_melville_data.jl")
include("plotting_utilities.jl")
dir = "../data"

cases = [
    "constant_waves_ep120_k30_beta120_N768_768_512_L20_20_10",
    "constant_waves_ep130_k30_beta120_N768_768_512_L20_20_10",
    "constant_waves_ep140_k30_beta120_N768_768_512_L20_20_10",
    #"sudden_waves_ep140_k30_beta120_N512_512_384_L20_20_10",
    #"sudden_waves_ep200_k30_beta120_N512_512_384_L20_20_10",
    #"sudden_waves_ep300_k30_beta120_N512_512_384_L20_20_10",
]

labels = [
    "constant waves with ϵ = 0.12",
    "constant waves with ϵ = 0.13",
    "constant waves with ϵ = 0.14",
    #"sudden waves with ϵ = 0.14",
    #"sudden waves with ϵ = 0.2",
    #"sudden waves with ϵ = 0.3",
]

linewidths = [6, 2, 2, 1]
t_transitions = [0, 0, 0, 0]
exp = "R2"
ramp = 2
t₀_udel = 79.7

udel_filename = joinpath(dir, "every_surface_velocity.mat")
udel_vars = matread(udel_filename)
u_udel = udel_vars["BIN"][exp]["U"][:]
t_udel = udel_vars["BIN"][exp]["time"][:] .- t₀_udel

# Figure
t₀ = 5
t₁ = 30
z₀ = -0.04
fig = Figure(resolution=(1200, 800))
colormap = :bilbao

ax_u = Axis(fig[1, 1], xaxisposition=:top,
            xlabel = "Simulation time (s)",
            ylabel = "Streamwise \n velocity (m s⁻¹)")

ax_w = Axis(fig[2, 1],
            xlabel = "Simulation time (s)",
            ylabel = "Cross-stream \n velocities (m s⁻¹)")

ylims!(ax_u, 0.0, 0.2)
xlims!(ax_u, t₀, t₁)

# ylims!(ax_w, 0.0, 0.2)
xlims!(ax_w, t₀, t₁)

# Scatter plot
surface_velocity_filename = joinpath(dir, "Final_SurfVel_per_RAMP.mat")
surface_velocity_data = matread(surface_velocity_filename)["BIN"]["R$ramp"]
t_surf = surface_velocity_data["time"][:]
u_surf = surface_velocity_data["U"][:]

u_surf_max = replace(u -> isnan(u) ? 0.0 : u, u_surf)
u_surf_max_max, n_max = findmax(u_surf_max)
t_surf_max = t_surf[n_max]
t_surf = t_surf .- t_surf_max

αvm = 0.5

scatter!(ax_u, t_udel, u_udel, marker=:circle, markersize=20, color=(:black, 0.5),
         label="Average surface velocity, UDelaware exp $exp")

#=
scatter!(ax_u, t_vm_surf, u_vm_surf, marker=:utriangle, markersize=20, color=(:blue, αvm),
         label="Surface velocity, Veron and Melville (2001)")

scatter!(ax_u, t_vm_avg_surf, u_vm_avg_surf, marker=:rect, markersize=20, color=(:purple, αvm),
         label="Average surface velocity, Veron and Melville (2001)")

scatter!(ax_u, t_vm_jet, u_vm_jet, markersize=20, color=(:indigo, αvm), marker=:dtriangle,
         label="Jet velocity, Veron and Melville (2001)")

scatter!(ax_u, t_vm_wake, u_vm_wake, marker=:cross, markersize=20, color=(:cyan, 1.0),
         label="Wake velocity, Veron and Melville (2001)")
=#

#####
##### Simulation stuff
#####

for n = 1:length(cases)

    case = cases[n]
    label = labels[n]
    αsim = 0.6
    statistics_filename = case * "_hi_freq_statistics.jld2"
    averages_filename   = case * "_hi_freq_averages.jld2"

    statistics_filepath = joinpath(dir, statistics_filename)
    averages_filepath   = joinpath(dir, averages_filename)

    U = FieldTimeSeries(averages_filepath, "u")
    Nz = U.grid.Nz
    u_avg = [U[n][1, 1, Nz] for n in 1:length(U.times)]
    t_avg = U.times

    stats = compute_timeseries(statistics_filepath)
    t_stats = stats[:t]
    u_max = stats[:u_max]
    w_max = stats[:w_max]
    v_max = stats[:v_max]
    u_min = stats[:u_min]

    # Shift time according to turbulent transition
    nn = sortperm(t_stats)
    t_stats = t_stats[nn]
    u_max = u_max[nn]
    v_max = v_max[nn]
    w_max = w_max[nn]

    #=
    Δ_max = 1e-6
    n_transition = findfirst(i -> u_max[i+1] < u_max[i] - Δ_max, 1:length(u_max)-1)
    isnothing(n_transition) && (n_transition=length(u_max))
    t_transition = t_stats[n_transition]
    =#

    t_transition = t_transitions[n]

    t_stats = t_stats .- t_transition
    t_avg = t_avg .- t_transition

    ct = FieldTimeSeries(averages_filepath, "c")
    c_sim = interior(ct, 1, 1, :, :)
    z_sim = znodes(ct.grid, Center())
    t_sim = ct.times .- t_transition
    Nt = length(t_sim)

    linewidth = linewidths[n]

    lines!(ax_u, t_stats, u_max; linewidth, color = (:darkred, 0.8), label =  "max(u), " * label)
    lines!(ax_u, t_stats, u_min; linewidth, color = (:seagreen, 0.6), label =  "min(u), " * label)
    lines!(ax_u, t_avg,   u_avg; linewidth, color = (:royalblue1, 0.6), label = "mean(u), " * label)

    lines!(ax_w, t_stats, v_max; linewidth, color = (:royalblue1, 0.8), label =  "max(v)")
    lines!(ax_w, t_stats, w_max; linewidth, color = (:darkred, 0.8), label =  "max(w)")
end

Legend(fig[0, 1], ax_u, tellwidth=false)
axislegend(ax_w, position=:lt)
hidespines!(ax_w, :t, :r)
hidespines!(ax_u, :b, :r)

display(fig)

save("surface_velocity_comparison.png", fig)
