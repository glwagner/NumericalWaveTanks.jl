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
fig = Figure(resolution=(1200, 1600))
colormap = :bilbao

ax_u    = Axis(fig[1, 1], xlabel="Time relative to turbulent transition (seconds)", ylabel="Maximum streamwise velocity (m s⁻¹)")
ax_w    = Axis(fig[2, 1], xlabel="Time relative to turbulent transition (seconds)", ylabel="Maximum vertical velocity (m s⁻¹)")
ax_sim  = Axis(fig[3, 1], xlabel="Time relative to turbulent transition (seconds)", ylabel="z (m)")

xlims!(ax_u,    t₀, t₁)
xlims!(ax_w,    t₀, t₁)
xlims!(ax_sim,  t₀, t₁)
ylims!(ax_sim,  z₀, 0.0)

# ax_llif = Axis(fig[2, 1], xlabel="Time relative to turbulent transition (seconds)", ylabel="z (m)")
#xlims!(ax_llif, t₀, t₁)
#ylims!(ax_llif, z₀, 0.0)

#=
lif_levels = collect(range(0, stop=500, length=6))
push!(lif_levels, maximum(c_llif))
contourf!(ax_llif, t_llif, z_llif, c_llif; colorrange=(50, 500), colormap, levels=lif_levels)
=#

c_max = 0.02
sim_levels = collect(range(0, stop=c_max, length=6))
push!(sim_levels, maximum(c_sim))

# Clip negative values
c_sim = max.(0, c_sim)

contourf!(ax_sim, t_sim, z_sim, c_sim'; colormap, colorrange=(0, c_max), levels=sim_levels)

# Scatter plot
surface_velocity_filename = "data/Final_SurfVel_per_RAMP.mat"
surface_velocity_data = matread(surface_velocity_filename)["BIN"]["R$ramp"]
t_surf = surface_velocity_data["time"][:]
u_surf = surface_velocity_data["U"][:]

u_surf_max = replace(u -> isnan(u) ? 0.0 : u, u_surf)
u_surf_max_max, n_max = findmax(u_surf_max)
t_surf_max = t_surf[n_max]
t_surf = t_surf .- t_surf_max

#scatter!(ax_u, t_surf, u_surf, marker=:pentagon, markersize=15, color=(:blue, 0.6), label="Lab (feature tracking)")

αsim = 0.6
αvm = 0.7

lines!(ax_u, t_stats, u_max, linewidth=6, color=(:red, αsim), label="max(u), simulation")
lines!(ax_u, t_stats, u_min, linewidth=6, color=(:orange, αsim), label="min(u), simulation")
lines!(ax_u, t_avg, u_avg, linewidth=4, color=(:black, αsim), label="mean(u), simulation")

lines!(ax_w, t_stats, w_max, linewidth=6, color=(:red, αsim), label="max(w), simulation")

scatter!(ax_u, t_vm_surf, u_vm_surf, marker=:utriangle, markersize=20, color=(:blue, αvm),
         label="Surface velocity, Veron and Melville (2001)")

scatter!(ax_u, t_vm_avg_surf, u_vm_avg_surf, marker=:rect, markersize=20, color=(:purple, αvm),
         label="Average surface velocity, Veron and Melville (2001)")

scatter!(ax_u, t_vm_jet, u_vm_jet, markersize=10, color=(:indigo, αvm),
         label="Jet velocity, Veron and Melville (2001)")

scatter!(ax_u, t_vm_wake, u_vm_wake, marker=:cross, markersize=20, color=(:cyan, 1.0),
         label="Wake velocity, Veron and Melville (2001)")

Legend(fig[0, 1], ax_u, tellwidth=false)

#=
tn_tlif = @lift t_tlif[$n] - 19.5 #- t_surf_max
vlines!(ax_u, tn_tlif, linestyle=:solid, linewidth=3, color=(:black, 0.6))
vlines!(ax_u, tn_sim, linestyle=:dash, linewidth=3, color=(:red, 0.8))
=#

display(fig)

save("/Users/gregorywagner/Desktop/figure_1.png", fig)
