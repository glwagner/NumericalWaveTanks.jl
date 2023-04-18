using Oceananigans
using GLMakie
using JLD2
using Statistics
using Printf

dir = "data"

filenames = [
    "increasing_wind_ep10_k20944_Nx384_Nz384_Lz2_xy_top.jld2",
    "increasing_wind_ep10_k30_N512_512_512_L20_20_10_xy_top.jld2",
]

descriptions = [
    "384³, Lz = 0.20 m",
    "512³, Lz = 0.20 m",
]

function compute_timeseries(name)
    file = jldopen(joinpath(dir, name))

    timeseries = Dict()
    for f in ("u", "v", "w", "c")
        timeseries[f] = FieldTimeSeries(joinpath(dir, name), f)
    end

    times = timeseries["u"].times
    Nt = length(times)
    max_u = zeros(Nt)
    ut = timeseries["u"]
    for n = 1:Nt
        max_u[n] = maximum(interior(ut[n]))
    end

    timeseries["max_u"] = max_u

    return timeseries
end

Nfiles = length(descriptions)
xy_timeseries = []
for f = 1:Nfiles
    push!(xy_timeseries, compute_timeseries(filenames[f]))
end

fig = Figure()
ax = Axis(fig[1, 1])

for f = 1:Nfiles
    ts = xy_timeseries[f]
    t = ts["u"].times
    max_u = ts["max_u"]
    lines!(ax, t, max_u, label=descriptions[f])
end

display(fig)

#=
times = first(xy_timeseries)["c"].times
Nt = length(times)

fig = Figure(resolution=(2400, 1200))

slider = Slider(fig[4, 1:Nfiles], range=1:Nt, startvalue=1)
n = slider.value

title = @lift string("Langmuir instability at t = ", prettytime(times[$n]))
Label(fig[1, 1:Nfiles], title)

for i = 1:Nfiles
    axcxy = Axis(fig[2, i], aspect=1, title=descriptions[i])
    axuxy = Axis(fig[3, i], aspect=1)
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

record(fig, "langmuir_surface.mp4", 1:Nt, framerate=50) do nn
    n[] = nn
end

=#
