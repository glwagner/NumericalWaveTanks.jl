using Oceananigans
using JLD2
using GLMakie

prefix = "weak_initial_turbulence_beta120_t016_N768_768_512_L20_20_10"
statistics_filename = prefix * "_hi_freq_statistics.jld2"
slice_filename = prefix * "_yz_left.jld2"

filepath = joinpath("..", "data", statistics_filename)
file = jldopen(filepath)
iterations = parse.(Int, keys(file["timeseries/t"]))
@show keys(file["timeseries"])
umax = [file["timeseries/u_max/$i"] for i in iterations]
vmax = [file["timeseries/v_max/$i"] for i in iterations]
wmax = [file["timeseries/w_max/$i"] for i in iterations]
t = [file["timeseries/t/$i"] for i in iterations]
close(file)

slice_filepath = joinpath("..", "data", slice_filename)
wt = FieldTimeSeries(slice_filepath, "w")

set_theme!(Theme(fontsize=24, linewidth=3))
fig = Figure()

ax = Axis(fig[1, 1], yscale=log10, xlabel="Time (s)", ylabel="Max velocity components",
          yticks=([1e-2, 1e-1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"]))

ylims!(ax, 1e-1, 1e1)

#xlims!(ax, t[2], t[end])
lines!(ax, t, umax ./ 0.16, label="max|u| z=0")
lines!(ax, t, vmax ./ 0.16, label="max|v| z=0")
lines!(ax, t, wmax ./ 0.16, label="max|w|")
axislegend(ax)

slice_times = wt.times
Nt_slice = length(slice_times)

#=
vt = FieldTimeSeries(slice_filepath, "v")
w = wt[Nt_slice]
v = vt[Nt_slice]
ξ = compute!(Field(∂y(w) - ∂z(v)))
ξn = interior(ξ, 1, :, :)
x, y, z = nodes(ξ)
=#

wn = interior(wt[Nt_slice], 1, :, :)
wlim = maximum(abs, wn) / 2
axw = Axis(fig[2, 1])
heatmap!(axw, y, z, wn, colormap=:balance, colorrange=(-wlim, wlim))
#heatmap!(axw, y, z, ξn)

display(fig)
