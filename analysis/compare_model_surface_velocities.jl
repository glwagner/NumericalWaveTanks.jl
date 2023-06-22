using GLMakie
using JLD2
using MAT
using Oceananigans

set_theme!(Theme(fontsize=24))

include("veron_melville_data.jl")
include("plotting_utilities.jl")
dir = "../data"

cases = [
    "constant_waves_medium_ic_ep80_k30_beta120_N768_768_512_L10_10_5",
    "constant_waves_medium_ic_ep100_k30_beta120_N768_768_512_L10_10_5",
    "constant_waves_medium_ic_ep110_k30_beta120_N768_768_512_L10_10_5",
]

labels = [
    "constant waves with ϵ = 0.08 and medium IC",
    "constant waves with ϵ = 0.10 and medium IC",
    "constant waves with ϵ = 0.11 and medium IC",
]

colors = Makie.wong_colors()
linewidths = [6, 6, 6, 1]
t_transitions = [0, 0, 0, 0]
exp = "R2"
ramp = 2
t₀_udel = 79.5

udel_filename = joinpath(dir, "every_surface_velocity.mat")
udel_vars = matread(udel_filename)
u_udel = udel_vars["BIN"][exp]["U"][:]
t_udel = udel_vars["BIN"][exp]["time"][:] .- t₀_udel

# Figure
t₀ = 15
t₁ = 23
z₀ = -0.04
fig = Figure(resolution=(1100, 1600))
colormap = :bilbao

ax_u = Axis(fig[2, 1], xaxisposition=:top,
            xlabel = "Simulation time (s)",
            ylabel = "Streamwise \n velocity (m s⁻¹)")

ax_w = Axis(fig[3, 1],
            yscale = log10,
            xlabel = "Simulation time (s)",
            ylabel = "Cross-stream \n velocities (m s⁻¹)")

xlims!(ax_u, t₀, t₁)
xlims!(ax_w, t₀, t₁)
ylims!(ax_u, 0.05, 0.2)

# Scatter plot
surface_velocity_filename = joinpath(dir, "Final_SurfVel_per_RAMP.mat")
surface_velocity_data = matread(surface_velocity_filename)["BIN"]["R$ramp"]
t_surf = surface_velocity_data["time"][:]
u_surf = surface_velocity_data["U"][:]

u_surf_max = replace(u -> isnan(u) ? 0.0 : u, u_surf)
u_surf_max_max, n_max = findmax(u_surf_max)
t_surf_max = t_surf[n_max]
t_surf = t_surf .- t_surf_max

scatter!(ax_u, t_udel, u_udel, marker=:circle, markersize=20, color=(:black, 0.5),
         label="Average surface velocity, UDelaware exp $exp")

#####
##### Simulation stuff
#####

for c = 1:length(cases)
    case = cases[c]
    label = labels[c]
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

    t_transition = t_transitions[c]
    t_stats = t_stats .- t_transition
    t_avg = t_avg .- t_transition

    ct = FieldTimeSeries(averages_filepath, "c")
    c_sim = interior(ct, 1, 1, :, :)
    z_sim = znodes(ct.grid, Center())
    t_sim = ct.times .- t_transition
    Nt = length(t_sim)

    linewidth = linewidths[c]

    #lines!(ax_u, t_stats, u_max; linewidth, color = (:darkred, 0.8), label =  "max(u), " * label)
    #lines!(ax_u, t_stats, u_min; linewidth, color = (:seagreen, 0.6), label =  "min(u), " * label)
    lines!(ax_u, t_avg,   u_avg; linewidth, color = colors[c], label = "mean(u), " * label)

    #lines!(ax_w, t_stats, v_max; linewidth, color = (:royalblue1, 0.8), label =  "max(v)")
    lines!(ax_w, t_stats, w_max; linewidth, color = colors[c], label =  "max|w|")
end

ut = []
wt = []
ct = []
for c in 1:2 #ength(cases)
    case = cases[c]
    slice_filename = case * "_yz_right.jld2"
    slice_filepath = joinpath(dir, slice_filename)
    wc = FieldTimeSeries(slice_filepath, "w")
    uc = FieldTimeSeries(slice_filepath, "u")
    cc = FieldTimeSeries(slice_filepath, "c")
    push!(wt, wc)
    push!(ut, uc)
    push!(ct, cc)
    times = wc.times
    Nt = length(times)
end

x, y, z = nodes(ut[1])

x = x .* 1e2
y = y .* 1e2
z = z .* 1e2
aspect = 10 / 1.5

ax_c1 = Axis(fig[1, 2]; aspect)
ax_c2 = Axis(fig[2, 2]; aspect)
slider = Slider(fig[4, 1:2], range=1:length(times), startvalue=1)
n = slider.value
w1 = @lift interior(wt[1][$n], 1, :, :)
w2 = @lift interior(wt[2][$n], 1, :, :)
u1 = @lift interior(ut[1][$n], 1, :, :)
u2 = @lift interior(ut[2][$n], 1, :, :)
c1 = @lift interior(ct[1][$n], 1, :, :)
c2 = @lift interior(ct[2][$n], 1, :, :)
tn = @lift times[$n]
#heatmap!(ax_c1, y, z, u1)
#heatmap!(ax_c2, y, z, u2)
heatmap!(ax_c1, y, z, c1, colormap=:bilbao)
heatmap!(ax_c2, y, z, c2, colormap=:bilbao)
vlines!(ax_u, tn)
vlines!(ax_w, tn)

#####
##### Load LIF data as "concentration"
#####

lif_filename = "../data/TRANSVERSE_STAT_RAMP2_LIF_final.mat"
lif_data = matread(lif_filename)["STAT_R2"]

c_lif = lif_data["LIFa"]
c_lif = permutedims(c_lif, (2, 3, 1)) # puts time in last dimension

# Load time (convert from 2D array to 1D vector)
t_lif = lif_data["time"][:] .- t₀_udel
x_lif = lif_data["X_transverse_m"][:]
z_lif = lif_data["Z_transverse_m"][:] .- 0.108
Nt = length(t_lif)

x_lif .-= minimum(x_lif)

# Convert to cm
x_lif .*= 1e2
z_lif .*= 1e2

colorrange = (100, 1500)
colormap = :bilbao
j1 = 1250
j2 = 1660

tn = @lift times[$n]
nlif = @lift searchsortedfirst(t_lif, $tn)
tn_lif = @lift begin
    t = t_lif[$nlif]
    @show t
    t
end
cn_lif = @lift rotr90(view(c_lif, :, :, $nlif))[:, j1:j2]

Lx = maximum(x_lif) - minimum(x_lif)
Lz = maximum(z_lif[j1:j2]) - minimum(z_lif[j1:j2])
# aspect = Lx / Lz

ax_lif = Axis(fig[3, 2]; xlabel="Cross-wind direction (cm)", ylabel="z (cm)", aspect)
heatmap!(ax_lif, x_lif, z_lif[j1:j2], cn_lif; colorrange, colormap=:bilbao)

Legend(fig[1, 1], ax_u, tellwidth=false, tellheight=false)
axislegend(ax_w, position=:lt)
hidespines!(ax_w, :t, :r)
hidespines!(ax_u, :b, :r)

xlims!(ax_c1,  0, 10)
xlims!(ax_c2,  0, 10)
xlims!(ax_lif, 0, 10)

ylims!(ax_c1,  -2.0, 0)
ylims!(ax_c2,  -2.0, 0)
ylims!(ax_lif, -2.0, 0)

colsize!(fig.layout, 1, Relative(0.2))

display(fig)

# save("compare_model_surface_velocities.png", fig)

