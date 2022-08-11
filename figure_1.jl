using GLMakie
using JLD2
using MAT
using Oceananigans

dir = "data"
#case = "increasing_wind_ep10_k30_N512_512_384_L20_20_10"
#case = "increasing_wind_ep27_k30_N384_384_256_L10_10_5"
case = "increasing_wind_ep28_k30_N384_384_256_L10_10_5"
#case = "increasing_wind_ep27_k30_N512_512_384_L10_10_5"
statistics_filename = case * "_statistics.jld2"
averages_filename   = case * "_averages.jld2"

statistics_filepath = joinpath(dir, statistics_filename)
averages_filepath   = joinpath(dir, averages_filename)

#####
##### Load LIF data
#####

ramp = "2"
llif_filename = "LIF_analysis_mean_profiles_ALL_EXPS.mat"
llif_data = matread(llif_filename)["STAT_R" * ramp * "_EXP2"]
t_llif = llif_data["time"][:] .- 98.8
c_llif = llif_data["mean_LIF"]
z_llif = -llif_data["depth_relative_to_surface"][:] .+ 0.001

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

stats = compute_timeseries(statistics_filepath)
t_stats = stats[:t]
u_max = stats[:u_max]
u_min = stats[:u_min]
n_transition = findfirst(i -> u_max[i+1] < u_max[i] - 1e-2, 1:length(u_max)-1)
t_transition = t_stats[n_transition]
t_stats = t_stats .- t_transition
@show t_transition

ct = FieldTimeSeries(averages_filepath, "c")
c_sim = interior(ct, 1, 1, :, :)
z_sim = znodes(Center, ct.grid)
t_sim = ct.times .- t_transition
Nt = length(t_sim)

# Figure
t₀ = -10
t₁ = 10
z₀ = -0.04
fig = Figure(resolution=(1100, 1200))
colormap = :bilbao

ax_u    = Axis(fig[1, 1], xlabel="Time relative to turbulent transition (seconds)", ylabel="Maximum surface velocity (m s⁻¹)")
ax_llif = Axis(fig[2, 1], xlabel="Time relative to turbulent transition (seconds)", ylabel="z (m)")
ax_sim  = Axis(fig[3, 1], xlabel="Time relative to turbulent transition (seconds)", ylabel="z (m)")

xlims!(ax_u,    t₀, t₁)
xlims!(ax_llif, t₀, t₁)
xlims!(ax_sim,  t₀, t₁)

ylims!(ax_llif, z₀, 0.0)
ylims!(ax_sim,  z₀, 0.0)

# Heatmaps
heatmap!(ax_llif, t_llif, z_llif, c_llif; colorrange=(50, 500), colormap)
heatmap!(ax_sim, t_sim, z_sim, c_sim'; colormap, colorrange=(0, 0.02))

# Scatter plot
surface_velocity_filename = "Final_SurfVel_per_RAMP.mat"
surface_velocity_data = matread(surface_velocity_filename)["BIN"]["R$ramp"]
t_surf = surface_velocity_data["time"][:]
u_surf = surface_velocity_data["U"][:]

u_surf_max = replace(u -> isnan(u) ? 0.0 : u, u_surf)
u_surf_max_max, n_max = findmax(u_surf_max)
t_surf_max = t_surf[n_max]
t_surf = t_surf .- t_surf_max

scatter!(ax_u, t_surf, u_surf, marker=:utriangle, markersize=15, color=(:blue, 0.6), label="Lab (feature tracking)")
scatter!(ax_u, t_stats, u_max, markersize=10, color=(:red, 0.6), label="max(u), simulation")
scatter!(ax_u, t_stats, u_min, markersize=10, color=(:orange, 0.6), label="min(u), simulation")
axislegend(ax_u, position=:rb)

#=
tn_tlif = @lift t_tlif[$n] - 19.5 #- t_surf_max
vlines!(ax_u, tn_tlif, linestyle=:solid, linewidth=3, color=(:black, 0.6))
vlines!(ax_u, tn_sim, linestyle=:dash, linewidth=3, color=(:red, 0.8))
=#

display(fig)

