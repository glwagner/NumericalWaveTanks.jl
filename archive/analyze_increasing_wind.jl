using Oceananigans
using JLD2
using GLMakie
using Printf
using Statistics

Nx = Ny = Nz = 256
dir = "increasing_wind"

function get_u_time_series(ϵ, k)
    prefix = @sprintf("%s_%d_%d_%d_k%.1e_ep%.1e", "increasing_wind", Nx, Ny, Nz, k, ϵ)
    λ = 2π / k

    xy_filepath = joinpath(dir, prefix * "_xy.jld2")
    statistics_filepath = joinpath(dir, prefix * "_statistics.jld2")

    statistics_file = jldopen(statistics_filepath)

    iterations = parse.(Int, keys(statistics_file["timeseries/t"]))
    t = [statistics_file["timeseries/t/$i"] for i in iterations]
    u_max = [statistics_file["timeseries/u_max/$i"] for i in iterations]
    v_max = [statistics_file["timeseries/v_max/$i"] for i in iterations]
    w_max = [statistics_file["timeseries/w_max/$i"] for i in iterations]
    u_min = [statistics_file["timeseries/u_min/$i"] for i in iterations]

    xy_file = jldopen(xy_filepath)

    iterations = parse.(Int, keys(xy_file["timeseries/t"]))
    t = [xy_file["timeseries/t/$i"] for i in iterations]
    u_xy = [xy_file["timeseries/u/$i"] for i in iterations]

    close(xy_file)
    u_avg = map(ui -> mean(ui), u_xy)

    return t, u_max, u_min, u_avg, v_max, w_max
end

ϵ = 0.3
k = 3.1e2
λ = 2π / k

t, u_max, u_min, u_avg, v_max, w_max = get_u_time_series(ϵ, k)

label = @sprintf("ϵ = %.1f, λ = %.1f cm", ϵ, 1e2 * λ)

theme = Theme(fontsize=24, linewidth=2)
set_theme!(theme)

fig = Figure(resolution=(1800, 1200))
ax = Axis(fig[1, 1]; xlabel = "Time (s)", ylabel = "u (cm s⁻¹)")
lines!(ax, t, 1e2 * u_max, label = "max(|u|), " * label)
lines!(ax, t, 1e2 * u_min, label = "min(|u|), " * label)
lines!(ax, t, 1e2 * u_avg, label = "⟨u⟩ˣʸ, " * label)

axislegend(ax; position=:rb)

ax_u = Axis(fig[2, 1]; xlabel = "Time (s)", ylabel = "u (cm s⁻¹)")
ax_v = Axis(fig[3, 1]; xlabel = "Time (s)", ylabel = "v (cm s⁻¹)")
ax_w = Axis(fig[4, 1]; xlabel = "Time (s)", ylabel = "w (cm s⁻¹)")

colorcycle = [:black, :red, :darkblue, :orange, :pink1, :seagreen, :magenta2]
i = 1
for ϵ in (0.1, 0.2, 0.3)
    #for k in (2.1e2, 3.1e2)
    
    k = 3.1e2
        λ = 2π / k
        t, u_max, u_min, u_avg, v_max, w_max = get_u_time_series(ϵ, k)
        label = @sprintf("ϵ = %.1f, λ = %.1f cm", ϵ, 1e2 * λ)
        lines!(ax_u, t, 1e2 * u_max, color = colorcycle[i], label = "max|u|, " * label)
        lines!(ax_v, t, 1e2 * v_max, color = colorcycle[i], label = "max|v|, " * label)
        lines!(ax_w, t, 1e2 * w_max, color = colorcycle[i], label = "max|w|, " * label)
        global i += 1
end

axislegend(ax_u; position=:rb)

display(fig)

