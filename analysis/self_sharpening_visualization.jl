using CairoMakie
using JLD2
using MAT
using Oceananigans

set_theme!(Theme(fontsize=32))

include("veron_melville_data.jl")
include("plotting_utilities.jl")
dir = "../data"

# Simulation case
#case = "constant_waves_ic000000_020000_ep120_k30_alpha120_N768_768_512_L10_10_5"
#case = "constant_waves_ic000000_020000_ep100_k30_alpha120_N768_768_512_L10_10_5"
case = "constant_waves_ic000000_050000_ep110_k30_alpha120_N768_768_512_L10_10_5"
#case = "constant_waves_ic000000_100000_ep110_k30_alpha120_N768_768_512_L10_10_5"

colors = Makie.wong_colors(0.7)
colormap = :bilbao
linewidth = 8
exp = "R2"
ramp = 2
t₀_udel = 79.5 # reference time for laboratory time-series

# Plot limits
u₀ = 0.08
u₁ = 0.2
w₀ = 5e-4
w₁ = 0.1
t₀ = 15
t₁ = 25
x₀ = 0
x₁ = 13.8
ylabelpadding = 15

# LIF parameters
lif_colorrange = (200, 1500)
sim_colorrange = (0, 0.1)
j1 = 1000
j2 = 1660

# Surface velocity time series
udel_filename = joinpath(dir, "every_surface_velocity.mat")
udel_vars = matread(udel_filename)
u_udel = udel_vars["BIN"][exp]["U"][:]
t_udel = udel_vars["BIN"][exp]["time"][:] .- t₀_udel

# LIF data
lif_filename = "../data/TRANSVERSE_STAT_RAMP2_LIF_final.mat"
lif_data = matread(lif_filename)["STAT_R2"]

c_lif = lif_data["LIFa"]
c_lif = permutedims(c_lif, (2, 3, 1)) # puts time in last dimension

# Load time (convert from 2D array to 1D vector)
t_lif = lif_data["time"][:] .- t₀_udel
x_lif = lif_data["X_transverse_m"][:]
z_lif = lif_data["Z_transverse_m"][:] .- 0.108
Nt = length(t_lif)

# Convert to cm
x_lif .-= minimum(x_lif)
x_lif .*= 1e2
z_lif .*= 1e3

Lx = maximum(x_lif) - minimum(x_lif)
Lz = maximum(z_lif[j1:j2]) - minimum(z_lif[j1:j2])

# Simulation data
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

slice_filename = case * "_yz_left.jld2"
slice_filepath = joinpath(dir, slice_filename)
ct = FieldTimeSeries(slice_filepath, "c")
times = ct.times
grid = ct.grid
Nt = length(times)
x_sim, y_sim, z_sim = nodes(ct)

x_sim = x_sim .* 1e2
y_sim = y_sim .* 1e2
z_sim = z_sim .* 1e3

#####
##### Figure
#####

fig = Figure(resolution=(2340, 1500))

# Surface velocity

xticks = [16, 18, 20, 22, 24]

ax_u = Axis(fig[1, 1]; xticks,
            xlabel = "Time (s)",
            ylabel = "Surface-averaged \n u (cm s⁻¹)")

ax_w = Axis(fig[1, 2]; xticks,
            yaxisposition = :right,
            yscale = log10,
            xlabel = "Time (s)",
            ylabel = "Surface maximum \n w (cm s⁻¹)")

scatter!(ax_u, t_udel, 100u_udel, marker=:circle, markersize=23, color=(:black, 0.6),
         label="Lab measurements")

lines!(ax_u, t_avg,  100u_avg; linewidth, color = colors[1], label = "Simulation with ϵ = 0.11, U' = 5 cm s⁻¹")
lines!(ax_w, t_stats, 100w_max; linewidth, color = colors[1], label = "max|w|")

Legend(fig[0, 1], ax_u, orientation = :horizontal)

text!(ax_u, 0.01, 0.04, text="(a)", space=:relative)
text!(ax_w, 0.01, 0.04, text="(b)", space=:relative)

xlims!(ax_u, t₀, t₁)
ylims!(ax_u, 100u₀, 100u₁)
xlims!(ax_w, t₀, t₁)
ylims!(ax_w, 100w₀, 100w₁)

# Heatmap

y_sim = vcat(y_sim, y_sim .+ 1e2 * grid.Ly)
z₀ = -12
z₁ = 1
aspect = 13.8 / 1.3

ax_sim = Axis(fig[2, 1]; aspect, xlabel="Cross-wind direction, y (cm)", xaxisposition=:top, ylabelpadding, ylabel="z (mm)")
ax_lif = Axis(fig[2, 2]; aspect, xlabel="Cross-wind direction, y (cm)", xaxisposition=:top, ylabelpadding, ylabel="z (mm)", yaxisposition=:right)
n = Observable(13)
tn_sim = @lift times[$n]

c_sim = @lift begin
    cn = interior(ct[$n], 1, :, :)
    vcat(cn, cn)
end

colors = Makie.wong_colors()
heatmap!(ax_sim, y_sim, z_sim, c_sim; colormap, colorrange=sim_colorrange)
vlines!(ax_u, tn_sim; linewidth=4, color=colors[2])
vlines!(ax_w, tn_sim; linewidth=4, color=colors[2])
text!(ax_u, @lift($tn_sim - 0.08), 9.000, text="(c, d)", color=colors[2], align=(:right, :bottom))
text!(ax_w, @lift($tn_sim - 0.08), 0.180, text="(c, d)", color=colors[2], align=(:right, :top))

nlif = @lift searchsortedfirst(t_lif, $tn_sim)
tn_lif = @lift t_lif[$nlif]
cn_lif = @lift rotr90(view(c_lif, :, :, $nlif))[:, j1:j2]
@show tn_lif

heatmap!(ax_lif, x_lif, z_lif[j1:j2], cn_lif; colorrange=lif_colorrange, colormap)

xlims!(ax_sim, x₀, x₁)
xlims!(ax_lif, x₀, x₁)
ylims!(ax_sim, z₀, z₁)
ylims!(ax_lif, z₀, z₁)
text!(ax_sim, 0.01, 0.04, text="(c)", color=colors[2], align=(:left, :bottom), space=:relative)
text!(ax_lif, 0.01, 0.04, text="(d)", color=colors[2], align=(:left, :bottom), space=:relative)

ax_sim = Axis(fig[3, 1]; aspect, ylabelpadding, ylabel="z (mm)")
ax_lif = Axis(fig[3, 2]; aspect, ylabelpadding, ylabel="z (mm)", yaxisposition=:right)
n = Observable(19)
tn_sim = @lift times[$n]

c_sim = @lift begin
    @show $n
    cn = interior(ct[$n], 1, :, :)
    vcat(cn, cn)
end

sim_colorrange = (0, 0.05)
heatmap!(ax_sim, y_sim, z_sim, c_sim; colormap, colorrange=sim_colorrange)
vlines!(ax_u, tn_sim; linewidth=4, color=colors[3])
vlines!(ax_w, tn_sim; linewidth=4, color=colors[3])
text!(ax_u, @lift($tn_sim - 0.08), 9.000, text="(e, f)", color=colors[3], align=(:right, :bottom))
text!(ax_w, @lift($tn_sim - 0.08), 0.200, text="(e, f)", color=colors[3], align=(:right, :top))

nlif = @lift searchsortedfirst(t_lif, $tn_sim)
tn_lif = @lift t_lif[$nlif]
cn_lif = @lift rotr90(view(c_lif, :, :, $nlif))[:, j1:j2]
@show tn_lif

heatmap!(ax_lif, x_lif, z_lif[j1:j2], cn_lif; colorrange=lif_colorrange, colormap)

xlims!(ax_sim, x₀, x₁)
xlims!(ax_lif, x₀, x₁)
ylims!(ax_sim, z₀, z₁)
ylims!(ax_lif, z₀, z₁)
hidexdecorations!(ax_sim)
hidexdecorations!(ax_lif)
text!(ax_sim, 0.01, 0.04, text="(e)", color=colors[3], align=(:left, :bottom), space=:relative)
text!(ax_lif, 0.01, 0.04, text="(f)", color=colors[3], align=(:left, :bottom), space=:relative)

z₀ = -16
z₁ = 1
aspect = 13.8 / 1.7

ax_sim = Axis(fig[4, 1]; aspect, ylabelpadding, ylabel="z (mm)")
ax_lif = Axis(fig[4, 2]; aspect, ylabelpadding, ylabel="z (mm)", yaxisposition=:right)
n = Observable(23)
tn_sim = @lift times[$n]

c_sim = @lift begin
    @show $n
    cn = interior(ct[$n], 1, :, :)
    vcat(cn, cn)
end

sim_colorrange = (0, 0.03)
heatmap!(ax_sim, y_sim, z_sim, c_sim; colormap, colorrange=sim_colorrange)
vlines!(ax_u, tn_sim; linewidth=4, color=colors[4])
vlines!(ax_w, tn_sim; linewidth=4, color=colors[4])
text!(ax_u, @lift($tn_sim - 0.06), 18.00, text="(g, h)", color=colors[4], align=(:right, :top), fontsize=30)
text!(ax_w, @lift($tn_sim - 0.06), 0.300, text="(g, h)", color=colors[4], align=(:right, :top), fontsize=30)

nlif = @lift searchsortedfirst(t_lif, $tn_sim)
tn_lif = @lift t_lif[$nlif]
cn_lif = @lift rotr90(view(c_lif, :, :, $nlif))[:, j1:j2]
@show tn_lif

heatmap!(ax_lif, x_lif, z_lif[j1:j2], cn_lif; colorrange=lif_colorrange, colormap)

xlims!(ax_sim, x₀, x₁)
xlims!(ax_lif, x₀, x₁)
ylims!(ax_sim, z₀, z₁)
ylims!(ax_lif, z₀, z₁)
hidexdecorations!(ax_sim)
hidexdecorations!(ax_lif)
text!(ax_sim, 0.01, 0.04, text="(g)", color=colors[4], align=(:left, :bottom), space=:relative)
text!(ax_lif, 0.01, 0.04, text="(h)", color=colors[4], align=(:left, :bottom), space=:relative)

z₀ = -24
z₁ = 1
aspect = 13.8 / 2.5

ax_sim = Axis(fig[5, 1]; aspect, ylabelpadding, ylabel="z (mm)")
ax_lif = Axis(fig[5, 2]; aspect, ylabelpadding, ylabel="z (mm)", yaxisposition=:right)
n = Observable(26)
tn_sim = @lift times[$n]

c_sim = @lift begin
    @show $n
    cn = interior(ct[$n], 1, :, :)
    vcat(cn, cn)
end

sim_colorrange = (0, 0.02)
heatmap!(ax_sim, y_sim, z_sim, c_sim; colormap, colorrange=sim_colorrange)
vlines!(ax_u, tn_sim; linewidth=4, color=colors[5])
vlines!(ax_w, tn_sim; linewidth=4, color=colors[5])
text!(ax_u, @lift($tn_sim + 0.08), 18.50, text="(i, j)", color=colors[5], align=(:left, :top))
text!(ax_w, @lift($tn_sim + 0.08), 0.400, text="(i, j)", color=colors[5], align=(:left, :center))

nlif = @lift searchsortedfirst(t_lif, $tn_sim)
tn_lif = @lift t_lif[$nlif]
cn_lif = @lift rotr90(view(c_lif, :, :, $nlif))[:, j1:j2]
@show tn_lif

lif_colorrange = (200, 700)
heatmap!(ax_lif, x_lif, z_lif[j1:j2], cn_lif; colorrange=lif_colorrange, colormap)

xlims!(ax_sim, x₀, x₁)
xlims!(ax_lif, x₀, x₁)
ylims!(ax_sim, z₀, z₁)
ylims!(ax_lif, z₀, z₁)
hidexdecorations!(ax_sim)
hidexdecorations!(ax_lif)
text!(ax_sim, 0.01, 0.04, text="(i)", color=colors[5], align=(:left, :bottom), space=:relative)
text!(ax_lif, 0.01, 0.04, text="(j)", color=colors[5], align=(:left, :bottom), space=:relative)

z₀ = -24
z₁ = 1
aspect = 13.8 / 2.5

ax_sim = Axis(fig[6, 1]; aspect, ylabelpadding, xlabel="Cross-wind direction, y (cm)", ylabel="z (mm)")
ax_lif = Axis(fig[6, 2]; aspect, ylabelpadding, xlabel="Cross-wind direction, y (cm)", ylabel="z (mm)", yaxisposition=:right)
#slider = Slider(fig[7, 1:2], range=1:length(times), startvalue=33)
#n = slider.value
n = Observable(31)
tn_sim = @lift times[$n]

c_sim = @lift begin
    @show $n
    cn = interior(ct[$n], 1, :, :)
    vcat(cn, cn)
end

sim_colorrange = (0, 0.012)
heatmap!(ax_sim, y_sim, z_sim, c_sim; colormap, colorrange=sim_colorrange)
vlines!(ax_u, tn_sim; linewidth=4, color=colors[6])
vlines!(ax_w, tn_sim; linewidth=4, color=colors[6])
text!(ax_u, @lift($tn_sim + 0.08), 19.00, text="(k, l)", color=colors[6], align=(:left, :top))
text!(ax_w, @lift($tn_sim + 0.08), 0.500, text="(k, l)", color=colors[6], align=(:left, :center))

nlif = @lift searchsortedfirst(t_lif, $tn_sim)
tn_lif = @lift t_lif[$nlif]
cn_lif = @lift rotr90(view(c_lif, :, :, $nlif))[:, j1:j2]
@show tn_lif

lif_colorrange = (200, 600)
heatmap!(ax_lif, x_lif, z_lif[j1:j2], cn_lif; colorrange=lif_colorrange, colormap)

xlims!(ax_sim, x₀, x₁)
xlims!(ax_lif, x₀, x₁)
ylims!(ax_sim, z₀, z₁)
ylims!(ax_lif, z₀, z₁)
text!(ax_sim, 0.01, 0.04, text="(k)", color=colors[6], align=(:left, :bottom), space=:relative)
text!(ax_lif, 0.01, 0.04, text="(l)", color=colors[6], align=(:left, :bottom), space=:relative)

rowsize!(fig.layout, 1, 200)
rowsize!(fig.layout, 4, 150)
rowsize!(fig.layout, 5, 200)
rowsize!(fig.layout, 6, 200)

display(fig)

save("self_sharpening_visualization.pdf", fig)

