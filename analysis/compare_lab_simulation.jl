using GLMakie
using JLD2
using MAT
using Oceananigans

set_theme!(Theme(fontsize=32))

dir = "../data"
case = "constant_waves_ep120_k30_beta120_N768_768_512_L20_20_10"

#yz_filename         = case * "_yz_left.jld2"
yz_filename         = case * "_yz_right.jld2"
statistics_filename = case * "_statistics.jld2"

yz_filepath = joinpath(dir, yz_filename)
statistics_filepath = joinpath(dir, statistics_filename)

#####
##### Load LIF data
#####

ramp = "2"
tlif_filename = string("../data/TRANSVERSE_STAT_RAMP", ramp, "_LIF_final.mat")
tlif_data = matread(tlif_filename)["STAT_R$ramp"]

# Load LIF data as "concentration"
c_tlif = tlif_data["LIFa"]
c_tlif = permutedims(c_tlif, (2, 3, 1)) # puts time in last dimension

# Load time (convert from 2D array to 1D vector)
t₀_udel = 79.5
t_tlif = tlif_data["time"][:]
t_tlif = t_tlif .- t₀_udel
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

c_sim = FieldTimeSeries(yz_filepath, "c", backend=OnDisk())
t_sim = c_sim.times
Nt = length(t_sim)

# Figure
fig = Figure(resolution=(2100, 400))
colormap = :bilbao

# LIF data heatmap
#
xticks = 0:0.02:0.12
ax_sim  = Axis(fig[1, 1]; xlabel="Transverse distance (m)", ylabel="Vertical distance (m)", xticks)
ax_tlif = Axis(fig[2, 1]; xlabel="Across-wind distance (m)", ylabel="Vertical distance (m)", xticks)

text!(ax_sim, 0.02, 0.02, space=:relative, text="(a)")
text!(ax_tlif, 0.02, 0.02, space=:relative, text="(b)")

hidexdecorations!(ax_sim)

j1 = 1200
j2 = 1600

x_tlif = tlif_data["X_transverse_m"][:]
z_tlif = tlif_data["Z_transverse_m"][j1:j2]
z_tlif = z_tlif .- maximum(z_tlif)

grid = c_sim.grid
Nx, Ny, Nz = size(grid)

# Lab data heatmap
n = Observable(151)
cn_tlif = @lift rotr90(view(c_tlif, :, :, $n))[:, j1:j2]

# colorrange=(50, 2500)
levels = 50:250:2500
contourf!(ax_tlif, x_tlif, z_tlif, cn_tlif; levels, extendhigh=:auto, colormap)

# Simulation heatmap
# n = 95
# n = searchsortedfirst(t_sim, tn_tlif) - 1
cn_sim = @lift begin
    m = searchsortedfirst(t_sim, t_tlif[$n])# - 1
    m = isnothing(m) ? length(t_sim) : m
    m = max(1, m)
    m = min(length(t_sim), m)
    @show t_sim[m]
    cn_sim = interior(c_sim[m], 1, :, :)
    cn_sim
end

# tn_sim = t_sim[n]
# @show tn_sim tn_tlif

x_sim, y_sim, z_sim = nodes(c_sim)

levels = 0.0:0.01:0.04
contourf!(ax_sim, y_sim, z_sim, cn_sim; levels, colormap, extendhigh=:auto)

z₀ = -0.015
xlims!(ax_tlif, 0, maximum(x_tlif))
ylims!(ax_tlif, z₀, 0)

xlims!(ax_sim, 0, maximum(x_tlif))
ylims!(ax_sim, z₀, 0)

display(fig)

# Nt = length(t_tlif)
# record(fig, "compare_lif_simulation.mp4", 1:Nt, framerate=12) do nn
#     @info "Plotting frame $nn of $Nt..."
#     n[] = nn
# end

#save("compare_lab_simulation.png", fig)
