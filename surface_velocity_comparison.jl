using GLMakie
using JLD2
using MAT
using Oceananigans

set_theme!(Theme(fontsize=24))

include("veron_melville_data.jl")
include("plotting_utilities.jl")

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

udel_filename = "data/every_surface_velocity.mat"
udel_vars = matread(udel_filename)
exp = "R2"

#t₀_udel = 95.9
t₀_udel = 96.3
u_udel = udel_vars["BIN"][exp]["U"][:]
t_udel = udel_vars["BIN"][exp]["time"][:] .- t₀_udel

t₀_vm = 19.0
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

ax_u = Axis(fig[1, 1],
            xlabel = "Time relative to turbulent transition (seconds)",
            ylabel = "Maximum streamwise velocity (m s⁻¹)")

ylims!(ax_u, 0.0, 0.2)
xlims!(ax_u, t₀, t₁)

# Scatter plot
surface_velocity_filename = "data/Final_SurfVel_per_RAMP.mat"
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

scatter!(ax_u, t_vm_surf, u_vm_surf, marker=:utriangle, markersize=20, color=(:blue, αvm),
         label="Surface velocity, Veron and Melville (2001)")

scatter!(ax_u, t_vm_avg_surf, u_vm_avg_surf, marker=:rect, markersize=20, color=(:purple, αvm),
         label="Average surface velocity, Veron and Melville (2001)")

scatter!(ax_u, t_vm_jet, u_vm_jet, markersize=20, color=(:indigo, αvm), marker=:dtriangle,
         label="Jet velocity, Veron and Melville (2001)")

scatter!(ax_u, t_vm_wake, u_vm_wake, marker=:cross, markersize=20, color=(:cyan, 1.0),
         label="Wake velocity, Veron and Melville (2001)")

#####
##### Simulation stuff
#####

dir = "data"

#case = "increasing_wind_ep14_k30_beta120_N384_384_256_L10_10_5_hi_freq"
#label = "ϵ = 0.14"
#case = "increasing_wind_ep135_k30_beta120_N384_384_256_L10_10_5_hi_freq"
#label = "ϵ = 0.135"
case = "increasing_wind_ep135_k30_beta120_N512_512_384_L10_10_5_hi_freq"
label = "ϵ = 0.13"
#case = "increasing_wind_ep13_k30_beta120_N512_512_384_L10_10_5_hi_freq"
#label = "ϵ = 0.13"
#case = "increasing_wind_ep12_k30_beta100_N512_512_384_L10_10_5_hi_freq"
#label = "ϵ = 0.12"
αsim = 0.6
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
w_max = stats[:w_max]
u_min = stats[:u_min]

# Shift time according to turbulent transition
nn = sortperm(t_stats)
t_stats = t_stats[nn]
u_max = u_max[nn]
w_max = w_max[nn]

Δ_max = 1e-6
n_transition = findfirst(i -> u_max[i+1] < u_max[i] - Δ_max, 1:length(u_max)-1)
isnothing(n_transition) && (n_transition=length(u_max))

t_transition = t_stats[n_transition]
t_stats = t_stats .- t_transition
t_avg = t_avg .- t_transition
@show length(u_max) n_transition t_transition

ct = FieldTimeSeries(averages_filepath, "c")
c_sim = interior(ct, 1, 1, :, :)
z_sim = znodes(Center, ct.grid)
t_sim = ct.times .- t_transition
Nt = length(t_sim)

lines!(ax_u, t_stats, u_max; linewidth=6, color = (:darkred, 0.8), label =  "max(u), " * label)
lines!(ax_u, t_stats, u_min; linewidth=6, color = (:seagreen, 0.6), label =  "min(u), " * label)
lines!(ax_u, t_avg,   u_avg; linewidth=8, color = (:royalblue1, 0.6), label = "mean(u), " * label)

Legend(fig[0, 1], ax_u, tellwidth=false)

display(fig)

save("surface_velocity_comparison.png", fig)
