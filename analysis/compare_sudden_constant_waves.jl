using GLMakie
using JLD2
using MAT
using Oceananigans

set_theme!(Theme(fontsize=24))

include("veron_melville_data.jl")
include("plotting_utilities.jl")
dir = "../data"

cases = [
    "constant_waves_ep140_k30_beta120_N512_512_384_L20_20_10",
    #"sudden_waves_ep140_k30_beta120_N512_512_384_L20_20_10",
    "sudden_waves_ep200_k30_beta120_N512_512_384_L20_20_10",
    "sudden_waves_ep300_k30_beta120_N512_512_384_L20_20_10",
]

labels = [
    "Constant waves with ϵ = 0.14",
    #"sudden waves with ϵ = 0.14",
    "Sudden waves with ϵ = 0.2",
    "Sudden waves with ϵ = 0.3",
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
t₀ = 10
t₁ = 25
z₀ = -0.04
fig = Figure(resolution=(1200, 800))
colormap = :bilbao

ax_u = Axis(fig[1, 1], xaxisposition=:top,
            xlabel = "Simulation time (s)",
            ylabel = "Streamwise \n surface \n velocity (m s⁻¹)")

ax_w = Axis(fig[2, 1],
            xlabel = "Simulation time (s)",
            ylabel = "Maximum \n cross-stream \n velocity (m s⁻¹)")

ylims!(ax_u, 0.1, 0.2)
xlims!(ax_u, t₀, t₁)

ylims!(ax_w, -0.01, 0.08)
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
         label="Lab average surface velocity")

scatter!(ax_w, [0.0], [0.0], marker=:circle, markersize=20, color=(:black, 0.5),
         label="Lab average surface velocity")

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

    linewidth = 4 #linewidths[n]

    lines!(ax_u, t_stats, u_max; linewidth, label)
    #lines!(ax_w, t_stats, w_max; linewidth)
    lines!(ax_w, t_stats, v_max; linewidth, label)
end

#Legend(fig[1:2, 2], ax_u, tellwidth=false)
axislegend(ax_w, position=:lt)
hidespines!(ax_w, :t, :r)
hidespines!(ax_u, :b, :r)

display(fig)

save("surface_velocity_sudden_waves.png", fig)

