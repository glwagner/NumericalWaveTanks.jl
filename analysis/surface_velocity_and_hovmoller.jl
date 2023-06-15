using GLMakie #CairoMakie
using JLD2
using MAT
using Oceananigans

set_theme!(Theme(fontsize=24))

include("veron_melville_data.jl")
include("plotting_utilities.jl")

dir = "../data"
#case = "constant_waves_ep140_k30_beta120_N768_768_512_L20_20_10"
case = "constant_waves_ep140_k30_beta750_N768_768_512_L20_20_10"
label = "constant waves with ϵ = 0.14"

linewidths = [6, 2, 2, 1]
t_transitions = [0, 0, 0, 0]
exp = "R2"
ramp = 2

t₀_udel = 12.4
udel_filename = joinpath(dir, "every_surface_velocity.mat")
udel_vars = matread(udel_filename)
u_udel = udel_vars["BIN"][exp]["U"][:]
t_udel = udel_vars["BIN"][exp]["time"][:] ./ 2π .- t₀_udel

# Figure
t₀ = 10
t₁ = 30
z₀ = -0.04
fig = Figure(resolution=(1200, 1200))
colormap = :bilbao

ax_u = Axis(fig[1, 1], xaxisposition=:top,
            xlabel = "Simulation time (s)",
            ylabel = "Along-wind \n velocity (m s⁻¹)")

#ylims!(ax_u, 0.0, 0.2)
#xlims!(ax_u, t₀, t₁)

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

#display(fig)

#####
##### Simulation stuff
#####

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

t_transition = 0.0 #t_transitions[n]
t_stats = t_stats .- t_transition
t_avg = t_avg .- t_transition

ct = FieldTimeSeries(averages_filepath, "c")
c_sim = interior(ct, 1, 1, :, :)
z_sim = znodes(ct.grid, Center())
t_sim = ct.times .- t_transition
Nt = length(t_sim)

linewidth = 4 #linewidths[n]

lines!(ax_u, t_stats, u_max; linewidth, color = (:darkred, 0.8), label =  "max(u), " * label)
lines!(ax_u, t_stats, u_min; linewidth, color = (:seagreen, 0.6), label =  "min(u), " * label)
lines!(ax_u, t_avg,   u_avg; linewidth, color = (:royalblue1, 0.6), label = "mean(u), " * label)

Legend(fig[0, 1], ax_u, tellwidth=false)
hidespines!(ax_u, :b, :r)

#####
##### Hovmoller
#####

new_filename = "../data/fig4b.mat"
ramp = "2"
t₀_udel = 79.7

lif_filename = "../data/fig4b.mat" #"data/LIF_analysis_mean_profiles_ALL_EXPS.mat"
lif_data = matread(lif_filename)["fig4b"]
t_lif = lif_data["time"][:] .- t₀_udel
c_lif = permutedims(lif_data["LIF"], (2, 1))
z_lif = -lif_data["z"][:] .+ 0.031

# Load simulation data
U = FieldTimeSeries(averages_filepath, "u")
Nz = U.grid.Nz
u_avg = [U[n][1, 1, Nz] for n in 1:length(U.times)]
t_avg = U.times

stats = compute_timeseries(statistics_filepath)
t_stats = stats[:t]
u_max = stats[:u_max]
w_max = stats[:w_max]
u_min = stats[:u_min]

ct = FieldTimeSeries(averages_filepath, "c")
c_sim = interior(ct, 1, 1, :, :)
c_sim = permutedims(c_sim, (2, 1))
z_sim = znodes(ct.grid, Center())
t_sim = ct.times
Nt = length(t_sim)

c_lif_min = minimum(c_lif)
c_lif = @. c_lif - c_lif_min
C_lif = sum(c_lif, dims=2)
c_lif = c_lif ./ C_lif[1]
ζk = mapslices(findmax, c_lif, dims=2)
ζ = map(elem -> elem[1], ζk)
K = map(elem -> elem[2], ζk)
# @show ζ K

zz = repeat(z_lif', length(t_lif), 1)
c_lif[c_lif .< 3e-4] .= 0
c_lif[zz .> 0.0001] .= 0

C_sim = sum(c_sim, dims=2)
C_lif = sum(c_lif, dims=2)

Z_sim = zeros(size(c_sim, 1))
Z_lif = zeros(size(c_lif, 1))

z95_sim = zeros(size(c_sim, 1))
z95_lif = zeros(size(c_lif, 1))

for n = 1:length(Z_sim)
    Z_sim[n] = sum(z_sim .* c_sim[n, :]) / C_sim[n]

    k = length(z_sim)
    C = 0.0
    while C < 0.95 * C_sim[n]
        C += c_sim[n, k]
        k -= 1
    end

    z95_sim[n] = z_sim[k] 
end

for n = 1:length(Z_lif)
    Z_lif[n] = sum(z_lif .* c_lif[n, :]) / C_lif[n]

    k = searchsortedfirst(z_lif, 0) - 1
    C = 0.0
    while C < 0.99999 * C_lif[n]
        C += c_lif[n, k]
        k -= 1
    end

    z95_lif[n] = z_lif[k] 
end

c_sim = c_sim ./ C_sim
c_sim = @. max(c_sim, 0, c_sim)

# Figure
z₀ = -0.08
z₁ = 0.001

colormap = :bilbao

yticks = -0.1:0.02:0.0
xticks = -0.1:0.02:0.0

ax_sim = Axis(fig[2, 1];
              xlabel = "Simulation time (s)",
              ylabel = "z (m)",
              yticks)

ax_lif = Axis(fig[3, 1];
              xlabel = "Simulation time (s)",
              ylabel = "z (m)",
              yticks)

hidexdecorations!(ax_sim)
heatmap!(ax_lif, t_lif, z_lif, log10.(c_lif); colorrange=(-4, -1.5), colormap) #, label="LIF concentration")
band!(ax_lif, t_sim, 0, z95_sim, color=(:royalblue, 0.1))
lines!(ax_lif, t_sim, z95_sim, color=(:royalblue, 0.4), linewidth=4, label="Simulated z₉₅")
axislegend(ax_lif, position=:lb)

# lines!(ax_lif, t_lif, Z_lif, label="Z(t) LIF", color=:seagreen, linewidth=6)
# lines!(ax_lif, t_sim, Z_sim, label="Z(t) Simulation", color=:royalblue, linewidth=4)

xlims!(ax_lif, t₀, t₁)
ylims!(ax_lif, z₀, z₁)

# Clip negative values
heatmap!(ax_sim, t_sim, z_sim, log10.(c_sim); colormap, colorrange=(-4, -2.5))

xlims!(ax_sim, t₀, t₁)
ylims!(ax_sim, z₀, z₁)

# xlims!(ax_95, t₀, t₁)
# ylims!(ax_95, z₀, z₁)

display(fig)

# save("surface_velocity_comparison.pdf", fig)
