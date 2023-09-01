using CairoMakie #GLMakie
using JLD2
using MAT
using Oceananigans

set_theme!(Theme(fontsize=32))

include("veron_melville_data.jl")
include("plotting_utilities.jl")
dir = "../data"

# Simulation case
cases = [
    "constant_waves_ic000000_050000_ep110_k30_alpha120_N768_768_512_L10_10_5",
    #"constant_waves_ic000000_050000_ep120_k30_alpha120_N768_768_512_L10_10_5",
    #"constant_waves_ic000000_100000_ep110_k30_alpha120_N768_768_512_L10_10_5",
    "constant_waves_ic000000_020000_ep110_k30_alpha120_N768_768_512_L10_10_5",
    "constant_waves_ic000000_050000_ep120_k30_alpha120_N768_768_512_L10_10_5",
    "constant_waves_ic000000_050000_ep100_k30_alpha120_N768_768_512_L10_10_5"
]

labels= [
    "ϵ = 0.11, U′ = 5 cm s⁻¹",
    #"ϵ = 0.12, U′ = 0.05",
    #"ϵ = 0.11, U′ = 10 cm s⁻¹",
    "ϵ = 0.11, U′ = 2 cm s⁻¹",
    "ϵ = 0.12, U′ = 5 cm s⁻¹",
    "ϵ = 0.10, U′ = 5 cm s⁻¹",
]

colors = Any[c for c in Makie.wong_colors(0.5)]
pushfirst!(colors, (:black, 1.0))
# colors[4] = (:black, 0.7)
linestyles = [:solid, :solid, :solid, :solid]
colormap = :bilbao
linewidth = 8
exp = "R2"
ramp = 2
t₀_udel = 79.5 # reference time for laboratory time-series

# Plot limits
u₀ = 0.08
u₁ = 0.2
w₀ = 5e-4
w₁ = 0.1
t₀ = 15
t₁ = 23
x₀ = 0
x₁ = 13.8
ylabelpadding = 15

# LIF parameters
lif_colorrange = (200, 1500)
sim_colorrange = (0, 0.1)
j1 = 1000
j2 = 1660

# Surface velocity time series
udel_filename = joinpath(dir, "every_surface_velocity.mat")
udel_vars = matread(udel_filename)
u_udel = udel_vars["BIN"][exp]["U"][:]
t_udel = udel_vars["BIN"][exp]["time"][:] .- t₀_udel

fig = Figure(resolution=(2000, 500))

xticks = [16, 18, 20, 22, 24]

ax_u = Axis(fig[1, 1]; xticks,
            xlabel = "Time (s)",
            ylabel = "Surface-averaged \n u (cm s⁻¹)")

ax_w = Axis(fig[1, 2]; xticks,
#            yaxisposition = :right,
            yscale = log10,
            xlabel = "Time (s)",
            ylabel = "Surface maximum \n w (cm s⁻¹)")

scatter!(ax_u, t_udel, 100u_udel, marker=:circle, markersize=23, color=(:black, 0.6),
         label="Lab measurements")

for (n, case) in enumerate(cases)
    # Simulation data
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
    
    label = labels[n]
    lines!(ax_u, t_avg, 100u_avg;   linewidth, linestyle=linestyles[n], color = colors[n], label)
    lines!(ax_w, t_stats, 100w_max; linewidth, linestyle=linestyles[n], color = colors[n], label)
end

text!(ax_u, 0.01, 0.04, text="(a)", space=:relative)
text!(ax_w, 0.01, 0.04, text="(b)", space=:relative)
Legend(fig[1, 3], ax_u)

xlims!(ax_u, t₀, t₁)
ylims!(ax_u, 100u₀, 100u₁)
xlims!(ax_w, t₀, t₁)
ylims!(ax_w, 60w₀, 100w₁)

display(fig)

save("compare_surface_velocities.pdf", fig)

