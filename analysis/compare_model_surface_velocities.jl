using GLMakie
using JLD2
using MAT
using Oceananigans

set_theme!(Theme(fontsize=24))

include("veron_melville_data.jl")
include("plotting_utilities.jl")
dir = "../data"

cases = [
    "constant_waves_ep100_k30_beta120_x1_N768_768_512_L20_20_10"
    "constant_waves_ep100_k30_beta120_x2_N768_768_512_L20_20_10"
]

labels = [
    "constant waves with ϵ = 0.10 and Χ = 10⁻¹",
    "constant waves with ϵ = 0.10 and Χ = 10⁻²",
]

colors = Makie.wong_colors()
linewidths = [6, 6, 2, 1]
t_transitions = [0, 0, 0, 0]
exp = "R2"
ramp = 2
t₀_udel = 79.5

udel_filename = joinpath(dir, "every_surface_velocity.mat")
udel_vars = matread(udel_filename)
u_udel = udel_vars["BIN"][exp]["U"][:]
t_udel = udel_vars["BIN"][exp]["time"][:] .- t₀_udel

# Figure
t₀ = 15
t₁ = 23
z₀ = -0.04
fig = Figure(resolution=(1200, 800))
colormap = :bilbao

ax_u = Axis(fig[1, 1], xaxisposition=:top,
            xlabel = "Simulation time (s)",
            ylabel = "Streamwise \n velocity (m s⁻¹)")

ax_w = Axis(fig[2, 1],
            yscale = log10,
            xlabel = "Simulation time (s)",
            ylabel = "Cross-stream \n velocities (m s⁻¹)")

ylims!(ax_u, 0.1, 0.2)
xlims!(ax_u, t₀, t₁)
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

scatter!(ax_u, t_udel, u_udel, marker=:circle, markersize=20, color=(:black, 0.5),
         label="Average surface velocity, UDelaware exp $exp")

#####
##### Simulation stuff
#####

for c = 1:length(cases)
    case = cases[c]
    label = labels[c]
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

    t_transition = t_transitions[c]
    t_stats = t_stats .- t_transition
    t_avg = t_avg .- t_transition

    ct = FieldTimeSeries(averages_filepath, "c")
    c_sim = interior(ct, 1, 1, :, :)
    z_sim = znodes(ct.grid, Center())
    t_sim = ct.times .- t_transition
    Nt = length(t_sim)

    linewidth = linewidths[c]

    #lines!(ax_u, t_stats, u_max; linewidth, color = (:darkred, 0.8), label =  "max(u), " * label)
    #lines!(ax_u, t_stats, u_min; linewidth, color = (:seagreen, 0.6), label =  "min(u), " * label)
    lines!(ax_u, t_avg,   u_avg; linewidth, color = colors[c], label = "mean(u), " * label)

    #lines!(ax_w, t_stats, v_max; linewidth, color = (:royalblue1, 0.8), label =  "max(v)")
    lines!(ax_w, t_stats, w_max; linewidth, color = colors[c], label =  "max|w|")
end

wt = []
for c in 1:length(cases)
    case = cases[c]
    slice_filename = case * "_yz_right.jld2"
    slice_filepath = joinpath(dir, slice_filename)
    wc = FieldTimeSeries(slice_filepath, "w")
    push!(wt, wc)
    times = wc.times
    Nt = length(times)
end

x, y, z = nodes(wt[1])

ax_c1 = Axis(fig[3, 1])
ax_c2 = Axis(fig[4, 1])
slider = Slider(fig[5, 1], range=1:Nt, startvalue=1)
n = slider.value
w1 = @lift interior(wt[1][$n], 1, :, :)
w2 = @lift interior(wt[2][$n], 1, :, :)
tn = @lift times[$n]
heatmap!(ax_c1, y, z, w1)
heatmap!(ax_c2, y, z, w2)
vlines!(ax_u, tn)
vlines!(ax_w, tn)

ylims!(ax_c1, -0.04, 0.0)
ylims!(ax_c2, -0.04, 0.0)

Legend(fig[0, 1], ax_u, tellwidth=false)
axislegend(ax_w, position=:lt)
hidespines!(ax_w, :t, :r)
hidespines!(ax_u, :b, :r)

display(fig)

# save("compare_model_surface_velocities.png", fig)
