#=
#using GLMakie
using JLD2
using MAT
using Oceananigans

set_theme!(Theme(fontsize=24))

dir = "data"
case = "increasing_wind_ep27_k30_N384_384_256_L10_10_5"

yz_filename         = case * "_yz_left.jld2"
statistics_filename = case * "_statistics.jld2"

yz_filepath = joinpath(dir, yz_filename)
statistics_filepath = joinpath(dir, statistics_filename)

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
n_transition = findfirst(i -> u_max[i+1] < u_max[i], 1:length(u_max)-1)
t_transition = t_stats[n_transition]
t_stats = t_stats .- t_transition

c_sim = FieldTimeSeries(yz_filepath, "c")
t_sim = c_sim.times
Nt = length(t_sim)
=#

# Figure
fig = Figure(resolution=(2000, 800))
colormap = :bilbao

# LIF data heatmap
# ax_u    = Axis(fig[1, 1], xlabel="Time relative to turbulent transition (seconds)", ylabel="Maximum surface velocity (m s⁻¹)")
ax_sim  = Axis(fig[1, 1], xlabel="Transverse distance (m)", ylabel="Vertical distance (m)")
ax_tlif = Axis(fig[2, 1], xlabel="Transverse distance (m)", ylabel="Vertical distance (m)")

#sim_slider  = Slider(fig[3, 1], range=1:Nt, startvalue=n_transition)
#tlif_slider = Slider(fig[5, 1], range=1:Nt_tlif, startvalue=151)

j1 = 1200
j2 = 1600

x_tlif = tlif_data["X_transverse_m"][:]
z_tlif = tlif_data["Z_transverse_m"][j1:j2]
z_tlif = z_tlif .- maximum(z_tlif)

xlims!(ax_sim, 0, maximum(x_tlif))
ylims!(ax_sim, minimum(z_tlif), 0)

#n = tlif_slider.value
#cn_tlif = @lift rotr90(view(c_tlif, :, :, $n))[:, j1:j2]
#tn_tlif = @lift t_tlif[$n] - 19.5 #- t_surf_max

n = 151
cn_tlif = rotr90(view(c_tlif, :, :, n))[:, j1:j2]
tn_tlif = t_tlif[n] - 19.5 #- t_surf_max
heatmap!(ax_tlif, x_tlif, z_tlif, cn_tlif; colorrange=(50, 2500), colormap)

# Simulation heatmap
grid = c_sim.grid
Nx, Ny, Nz = size(grid)
#n = sim_slider.value
#tn_sim = @lift t_sim[$n] - t_transition
#cn_sim = @lift begin
#    c = interior(c_sim[$n], 1, :, :)
#    vcat(c, c)
#end

n = 64
tn_sim = t_sim[n] - t_transition
cn_sim = interior(c_sim[n], 1, :, :)
cn_sim = vcat(cn_sim, cn_sim)

x_sim, y_sim, z_sim = nodes(c_sim)
y_sim = range(0.0, stop=2y_sim[Ny], length=2Ny)

heatmap!(ax_sim, y_sim, z_sim, cn_sim; colorrange=(0.0001, 0.1), colormap)

# Scatter plot
surface_velocity_filename = "Final_SurfVel_per_RAMP.mat"
surface_velocity_data = matread(surface_velocity_filename)["BIN"]["R$ramp"]
t_surf = surface_velocity_data["time"][:]
u_surf = surface_velocity_data["U"][:]

u_surf_max = replace(u -> isnan(u) ? 0.0 : u, u_surf)
u_surf_max_max, n_max = findmax(u_surf_max)
t_surf_max = t_surf[n_max]
t_surf = t_surf .- t_surf_max

# scatter!(ax_u, t_surf, u_surf, marker=:circle)
# scatter!(ax_u, t_stats, u_max)
# vlines!(ax_u, tn_tlif, linestyle=:solid, linewidth=3, color=(:black, 0.6))
# vlines!(ax_u, tn_sim, linestyle=:dash, linewidth=3, color=(:red, 0.8))

display(fig)

