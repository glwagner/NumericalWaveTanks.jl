#=
using GLMakie
using JLD2
using MAT
using Oceananigans

dir = "data"
yz_filename = "increasing_wind_ep10_k30_N512_512_384_L20_20_10_yz_left.jld2"
statistics_filename = "increasing_wind_ep10_k30_N512_512_384_L20_20_10_statistics.jld2"
yz_filepath = joinpath(dir, yz_filename)
=#

#####
##### Load LIF data
#####

ramp = "2"
tlif_filename = string("TRANSVERSE_STAT_RAMP", ramp, "_LIF_final.mat")
tlif_data = matread(tlif_filename)["STAT_R$ramp"]

# Load LIF data as "concentration"
c_tlif = tlif_data["LIFa"]
c_tlif = permutedims(c_tlif, (2, 3, 1)) # puts time in last dimension

# Load time (convert from 2D array to 1D vector)
t_tlif = tlif_data["time"][:]
t_tlif = t_tlif .- t_tlif[1]
Nt_tlif = length(t_tlif)

llif_filename = "LIF_analysis_mean_profiles_ALL_EXPS.mat"
llif_data = matread(llif_filename)["STAT_R2_EXP1"]
t_llif = llif_data["time"]
c_llif = llif_data["mean_LIF"]

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
u_max_max, n_max = findmax(u_max)
t_max = t_stats[n_max]
t_stats = t_stats .- t_max

c_sim = FieldTimeSeries(yz_filepath, "c")
t_sim = c_sim.times
Nt = length(t_sim)

# Figure
fig = Figure(resolution=(2000, 1200))
colormap = :bilbao

ax_llif = Axis(fig[2, 1])
heatmap!(ax_llif, c_llif)
# Scatter plot
ax_u = Axis(fig[1, 1])

surface_velocity_data = matread(filename)["BIN"]["R$ramp"]
t_surf = surface_velocity_data["time"][:]
u_surf = surface_velocity_data["U"][:]

u_surf_max = replace(u -> isnan(u) ? 0.0 : u, u_surf)
u_surf_max_max, n_max = findmax(u_surf_max)
t_surf_max = t_surf[n_max]
t_surf = t_surf .- t_surf_max

scatter!(ax_u, t_surf, u_surf, marker=:circle)
scatter!(ax_u, t_stats, u_max)

tn_tlif = @tlift t_tlif[$n] - 19.5 #- t_surf_max
vlines!(ax_u, tn_tlif, linestyle=:solid, linewidth=3, color=(:black, 0.6))
vlines!(ax_u, tn_sim, linestyle=:dash, linewidth=3, color=(:red, 0.8))

display(fig)

