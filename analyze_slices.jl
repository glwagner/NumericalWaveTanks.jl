using Oceananigans
using GLMakie
using JLD2
using Statistics
using Printf

dir = "data"

prefixes = [
    "increasing_wind_ep10_k30_N256_256_256_L20_20_10",
    "increasing_wind_ep10_k30_N384_384_256_L20_20_10",
    "increasing_wind_ep10_k30_N384_384_384_L20_20_10",
    "increasing_wind_ep10_k30_N512_512_384_L20_20_10",
]

descriptions = [
    "256³,       Lz = 0.10 m",
    "384² × 256, Lz = 0.10 m",
    "384³,       Lz = 0.10 m",
    "512² × 384, Lz = 0.10 m",
]

yz_filenames = [p * "_yz_left.jld2" for p in prefixes]
xy_filenames = [p * "_xy_top.jld2" for p in prefixes]

function compute_timeseries(name)
    file = jldopen(joinpath(dir, name))

    timeseries = Dict()
    for f in ("u", "v", "w", "c")
        timeseries[f] = FieldTimeSeries(joinpath(dir, name), f)
    end

    return timeseries
end

Nfiles = length(descriptions)
yz_timeseries = []
xy_timeseries = []
for f = 1:Nfiles
    push!(yz_timeseries, compute_timeseries(yz_filenames[f]))
    push!(xy_timeseries, compute_timeseries(xy_filenames[f]))
end

times = first(yz_timeseries)["c"].times
Nt = length(times)

fig = Figure(resolution=(2400, 1200))

slider = Slider(fig[7, 1:Nfiles], range=1:Nt, startvalue=1)
n = slider.value

title = @lift string("Langmuir instability at t = ", prettytime(times[$n]))
Label(fig[1, 1:Nfiles], title)

for i = 1:Nfiles
    axcyz = Axis(fig[2, i], aspect=2, title=descriptions[i])
    axuyz = Axis(fig[3, i], aspect=2)
    yzts = yz_timeseries[i]
    c = yzts["c"]
    u = yzts["u"]
    grid = u.grid
    @show descriptions[i]
    @show grid
    x, y, z = nodes(c)
    cⁿyz = @lift interior(c[$n], 1, :, :)
    uⁿyz = @lift interior(u[$n], 1, :, :)
    heatmap!(axcyz, y, z, cⁿyz)
    heatmap!(axuyz, y, z, uⁿyz, colormap=:solar, colorrange=(-0.01, 0.1))
    ylims!(axcyz, -0.1, 0.0)
    xlims!(axcyz,  0.0, 0.2)
    ylims!(axuyz, -0.1, 0.0)
    xlims!(axuyz,  0.0, 0.2)
end

for i = 1:Nfiles
    axcxy = Axis(fig[4:5, i], aspect=1, title=descriptions[i])
    axuxy = Axis(fig[5:6, i], aspect=1)
    xyts = xy_timeseries[i]
    c = xyts["c"]
    u = xyts["u"]
    x, y, z = nodes(c)
    cⁿxy = @lift interior(c[$n], :, :, 1)
    uⁿxy = @lift interior(u[$n], :, :, 1)
    heatmap!(axcxy, x, y, cⁿxy)
    heatmap!(axuxy, x, y, uⁿxy, colormap=:solar, colorrange=(-0.01, 0.1))
    ylims!(axcxy,  0.0, 0.2)
    xlims!(axcxy,  0.0, 0.2)
    ylims!(axuxy,  0.0, 0.2)
    xlims!(axuxy,  0.0, 0.2)
end

display(fig)

record(fig, "langmuir_slices.mp4", 1:Nt, framerate=50) do nn
    n[] = nn
end

