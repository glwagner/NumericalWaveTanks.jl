using Oceananigans
using GLMakie
using JLD2
using Statistics
using Printf

dir = "data"

prefixes = [
    "increasing_wind_ep10_k30_beta120_N384_384_256_L10_10_5",
    "increasing_wind_ep14_k30_beta120_N384_384_256_L10_10_5",
    "increasing_wind_ep18_k30_beta120_N384_384_256_L10_10_5",
]

statistics_filenames = [prefix * "_hi_freq_statistics.jld2" for prefix in prefixes]
averages_filenames = [prefix * "_hi_freq_averages.jld2" for prefix in prefixes]

function compute_timeseries(statsname, avgsname)
    statsfile = jldopen(joinpath(dir, statsname))

    iters = parse.(Int, keys(statsfile["timeseries/t"]))
    
    timeseries = Dict()
    timeseries[:filename] = statsname
    timeseries[:grid] = statsfile["serialized/grid"]

    for stat in (:u_max, :u_min, :v_max, :w_max, :t)
        timeseries[stat] = [statsfile["timeseries/$stat/$i"] for i in iters]
    end

    #=
    for avg in (:c, :u, :η²)
        timeseries[avg] = FieldTimeSeries(joinpath(dir, avgsname), string(avg))

        # volume averaged
        volmean = Symbol(avg, :_mean)
        avg = timeseries[avg]
        times = avg.times
        #timeseries[volmean] = [mean(avg[n]) for n = 1:length(times)]
    end
    =#

    close(statsfile)

    return timeseries
end

Nfiles = length(statistics_filenames)

timeseries = []
for f = 1:Nfiles
    push!(timeseries, compute_timeseries(statistics_filenames[f], averages_filenames[f]))
end

fig = Figure()
ax = Axis(fig[1, 1])

for f = 1:Nfiles
    ts = timeseries[f]
    t = ts[:t]
    umax = ts[:u_max]
    grid = ts[:grid]
    Nx, Ny, Nz = size(grid)
    Lz = grid.Lz
    
    label = @sprintf("Increasing wind (%d, %d, %d) with Lz = %.2f", Nx, Ny, Nz, Lz)
    lines!(ax, t, umax; label)
end

axislegend(ax, position=:rb)

display(fig)

