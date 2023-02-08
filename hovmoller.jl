using GLMakie
using JLD2
using MAT
using Oceananigans

set_theme!(Theme(fontsize=24))

include("veron_melville_data.jl")
include("plotting_utilities.jl")

dir = "data"
case = "increasing_wind_ep14_k30_beta120_N384_384_256_L10_10_5_hi_freq"
statistics_filename = case * "_statistics.jld2"
averages_filename   = case * "_averages.jld2"
Δ_max = 1e-3

statistics_filepath = joinpath(dir, statistics_filename)
averages_filepath   = joinpath(dir, averages_filename)

#####
##### Load LIF data
#####

ramp = "2"
llif_filename = "data/LIF_analysis_mean_profiles_ALL_EXPS.mat"
llif_data = matread(llif_filename)["STAT_R" * ramp * "_EXP2"]
t_llif = llif_data["time"][:] .- 98.8
c_llif = llif_data["mean_LIF"]
z_llif = -llif_data["depth_relative_to_surface"][:] .+ 0.001

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
#u_surf = u_surf[nn]
#u_jet = u_jet[nn]
#u_avg_surf = u_avg_surf[nn]
#u_wake = u_wake[nn]

n_transition = findfirst(i -> u_max[i+1] < u_max[i] - Δ_max, 1:length(u_max)-1)
t_transition = t_stats[n_transition]
t_stats = t_stats .- t_transition
t_avg = t_avg .- t_transition
@show t_transition

ct = FieldTimeSeries(averages_filepath, "c")
c_sim = interior(ct, 1, 1, :, :)
z_sim = znodes(Center, ct.grid)
t_sim = ct.times .- t_transition
Nt = length(t_sim)

t₀_vm = 19.4
t_vm_surf = veron_melville_data[:t_surf] .- t₀_vm
u_vm_surf = veron_melville_data[:u_surf] ./ 100

t_vm_avg_surf = veron_melville_data[:t_avg_surf] .- t₀_vm
u_vm_avg_surf = veron_melville_data[:u_avg_surf] ./ 100

t_vm_jet = veron_melville_data[:t_jet] .- t₀_vm
u_vm_jet = veron_melville_data[:u_jet] ./ 100

t_vm_wake = veron_melville_data[:t_wake] .- t₀_vm
u_vm_wake = veron_melville_data[:u_wake] ./ 100

# Figure
t₀ = -10
t₁ = 10
z₀ = -0.04
fig = Figure(resolution=(1200, 800))
colormap = :bilbao

ax_sim  = Axis(fig[1, 1], xlabel="Time relative to turbulent transition (seconds)", ylabel="z (m)")

xlims!(ax_sim,  t₀, t₁)
ylims!(ax_sim,  z₀, 0.0)

ax_llif = Axis(fig[2, 1], xlabel="Time relative to turbulent transition (seconds)", ylabel="z (m)")
xlims!(ax_llif, t₀, t₁)
ylims!(ax_llif, z₀, 0.0)

lif_levels = collect(range(0, stop=500, length=6))
push!(lif_levels, maximum(c_llif))
contourf!(ax_llif, t_llif, z_llif, c_llif; colorrange=(50, 500), colormap, levels=lif_levels)

c_max = 0.02
sim_levels = collect(range(0, stop=c_max, length=6))
push!(sim_levels, maximum(c_sim))

# Clip negative values
c_sim = max.(0, c_sim)

contourf!(ax_sim, t_sim, z_sim, c_sim'; colormap, colorrange=(0, c_max), levels=sim_levels)

#tn_tlif = @lift t_tlif[$n] - 19.5 #- t_surf_max
#vlines!(ax_u, tn_tlif, linestyle=:solid, linewidth=3, color=(:black, 0.6))
#vlines!(ax_u, tn_sim, linestyle=:dash, linewidth=3, color=(:red, 0.8))

display(fig)

save("hovmoller.png", fig)
