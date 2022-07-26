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

# LIF data heatmap
ax_tlif = Axis(fig[6, 1])
tlif_slider = Slider(fig[7, 1], range=1:Nt_tlif, startvalue=151)
n = tlif_slider.value

j1 = 1200
j2 = 1600

x_tlif = tlif_data["X_transverse_m"][:]
z_tlif = tlif_data["Z_transverse_m"][j1:j2]
z_tlif = z_tlif .- maximum(z_tlif)

cn_tlif = @tlift rotr90(view(c_tlif, :, :, $n))[:, j1:j2]
heatmap!(ax_tlif, x_tlif, z_tlif, cn_tlif; colorrange=(50, 2500), colormap)

# Simulation heatmap
ax_sim = Axis(fig[4, 1])
sim_slider = Slider(fig[5, 1], range=1:Nt, startvalue=n_max)
n = sim_slider.value

grid = c_sim.grid
Nx, Ny, Nz = size(grid)
Nh = findfirst(x -> x > maximum(x_tlif), xnodes(Face, grid))
@show Nh
Nv = 240
tn_sim = @tlift t_sim[$n] - t_max
cn_sim = @tlift interior(c_sim[$n], 1, 1:Nh, Nv:Nz)

x_sim, y_sim, z_sim = nodes(c_sim)
heatmap!(ax_sim, y_sim[1:Nh], z_sim[Nv:Nz], cn_sim; colorrange=(0.01, 0.065), colormap)

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

# title = @tlift string("Langmuir instability at t = ", prettytime(t[$n]))
# Label(fig[1, 1:2], title, tellwidth=false)

display(fig)

