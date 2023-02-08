using JLD2

function extract_slices(filepath; dims, name)
    file = jldopen(filepath)
    iters = parse.(Int, keys(file["timeseries/t"]))
    times = [file["timeseries/t/$i"] for i in iters]
    times, iters = clean_times_iters(times, iters)

    slices = [dropdims(file["timeseries/$name/$i"]; dims) for i in iters]
    close(file)

    return slices, times
end

function compute_timeseries(filepath)
    statsfile = jldopen(filepath)

    timeseries = Dict()
    timeseries[:filepath] = filepath
    timeseries[:grid] = statsfile["serialized/grid"]

    iters = parse.(Int, keys(statsfile["timeseries/t"]))
    times = [statsfile["timeseries/t/$i"] for i in iters]

    # Sort and remove near-duplicate entries
    times, iters = clean_times_iters(times, iters)
    timeseries[:t] = times

    for stat in (:u_max, :u_min, :v_max, :w_max)
        timeseries[stat] = [statsfile["timeseries/$stat/$i"] for i in iters]
    end

    close(statsfile)

    return timeseries
end

function clean_times_iters(times, iters; tol=1e-15)
    nn = sortperm(times)
    times = times[nn]
    iters = iters[nn]

    # Assume that the first time-step isn't messed up
    Δt = times[2] - times[1]
    
    filtered_times = Float64[]
    filtered_iters = Int[]

    for (n, iter) in enumerate(iters)
        if n == 1 || (times[n] - times[n-1]) >= Δt - tol
            push!(filtered_times, times[n])
            push!(filtered_iters, iter)
        end
    end

    return filtered_times, filtered_iters
end
