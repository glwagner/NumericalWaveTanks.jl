using GLMakie
using JLD2
using MAT
using Oceananigans

set_theme!(Theme(fontsize=24))

include("veron_melville_data.jl")
include("plotting_utilities.jl")

dir = "data"
case = "increasing_wind_ep135_k30_beta120_N512_512_384_L10_10_5"
statistics_filename = case * "_statistics.jld2"
averages_filename   = case * "_averages.jld2"
Δ_max = 1e-3

statistics_filepath = joinpath(dir, statistics_filename)
averages_filepath   = joinpath(dir, averages_filename)

#####
##### Load LIF data
#####

new_filename = "data/fig4b.mat"

ramp = "2"
lif_filename = "data/fig4b.mat" #"data/LIF_analysis_mean_profiles_ALL_EXPS.mat"
#lif_data = matread(lif_filename)["STAT_R" * ramp * "_EXP2"]
lif_data = matread(lif_filename)["fig4b"]
t_lif = lif_data["time"][:]
t_lif = t_lif .- t_lif[1]
t_lif = t_lif .- 4 # 4.83 #.- 98.8
c_lif = permutedims(lif_data["LIF"], (2, 1))
z_lif = -lif_data["z"][:] .+ 0.034

#####
##### Load simulation data
#####

U = FieldTimeSeries(averages_filepath, "u")
Nz = U.grid.Nz
u_avg = [U[n][1, 1, Nz] for n in 1:length(U.times)]
t_avg = U.times

stats = compute_timeseries(statistics_filepath)
t_stats = stats[:t]
u_max = stats[:u_max]
w_max = stats[:w_max]
u_min = stats[:u_min]

# Shift time according to turbulent transition
nn = sortperm(t_stats)
t_stats = t_stats[nn]
u_max = u_max[nn]
w_max = w_max[nn]

n_transition = findfirst(i -> u_max[i+1] < u_max[i] - Δ_max, 1:length(u_max)-1)
t_transition = t_stats[n_transition]

ct = FieldTimeSeries(averages_filepath, "c")
c_sim = interior(ct, 1, 1, :, :)
z_sim = znodes(Center, ct.grid)
t_sim = ct.times #.- t_transition
Nt = length(t_sim)

c_sim = permutedims(c_sim, (2, 1))
c_lif_min = minimum(c_lif)
c_lif = @. c_lif - c_lif_min
C_lif = sum(c_lif, dims=2)
c_lif = c_lif ./ C_lif[1]

c_lif[c_lif .< 3e-4] .= 0

C_sim = sum(c_sim, dims=2)
C_lif = sum(c_lif, dims=2)

Z_sim = zeros(size(c_sim, 1))
Z_lif = zeros(size(c_lif, 1))

for n = 1:length(Z_sim)
    Z_sim[n] = sum(z_sim .* c_sim[n, :]) / C_sim[n]
end

for n = 1:length(Z_lif)
    Z_lif[n] = sum(z_lif .* c_lif[n, :]) / C_lif[n]
end

c_sim = c_sim ./ C_sim

c_sim = @. max(c_sim, 0, c_sim)

# Figure
t₀ = 15.0 #-10
t₁ = 30 #^022.6 #12.6
z₀ = -0.05
z₁ = 0.02
fig = Figure(resolution=(1600, 1200))
colormap = :bilbao

yticks = -0.1:0.02:0.0
xticks = -0.1:0.02:0.0
ax_int = Axis(fig[1, 1]; xlabel="Time relative to turbulent transition (seconds)", ylabel="Normalized ∫c dz")
ax_cen = Axis(fig[2, 1]; xlabel="Time relative to turbulent transition (seconds)", ylabel="Z(t) = ∫z c dz")
ax_sim = Axis(fig[3, 1]; title="Simulation", xlabel="Time relative to turbulent transition (seconds)", ylabel="z (m)", yticks)
ax_lif = Axis(fig[4, 1]; title="Laboratory LIF", xlabel="Time relative to turbulent transition (seconds)", ylabel="z (m)", yticks)

lines!(ax_int, t_sim, C_sim[:] / C_sim[1], label="Simulation")
lines!(ax_int, t_lif, C_lif[:] / C_lif[1], label="LIF")

lines!(ax_cen, t_sim, Z_sim, label="Simulation")
lines!(ax_cen, t_lif, Z_lif, label="LIF")

axislegend(ax_cen, position=:lb)

heatmap!(ax_lif, t_lif, z_lif, log10.(c_lif); colorrange=(-4, -2), colormap)
lines!(ax_lif, t_lif, Z_lif, label="Simulation", color=:lightblue1, linewidth=6)

xlims!(ax_int, t₀, t₁)
xlims!(ax_cen, t₀, t₁)
xlims!(ax_lif, t₀, t₁)
ylims!(ax_lif, z₀, z₁)

# Clip negative values
heatmap!(ax_sim, t_sim, z_sim, log10.(c_sim); colormap, colorrange=(-4, -2.5))
lines!(ax_sim, t_sim, Z_sim, label="Simulation", color=:lightblue1, linewidth=6)

xlims!(ax_sim, t₀, t₁)
ylims!(ax_sim, z₀, z₁)

display(fig)

# save("hovmoller.png", fig)
