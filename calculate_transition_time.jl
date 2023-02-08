using Oceananigans
using GLMakie
using Printf
using Statistics

include("veron_melville_data.jl")
include("plotting_utilities.jl")

set_theme!(Theme(fontsize=36))

dir = "data"

#=
filenames = [
    "increasing_wind_ep12_k30_beta110_N384_384_256_L10_10_5_hi_freq_statistics.jld2",  
    "increasing_wind_ep14_k30_beta110_N384_384_256_L10_10_5_hi_freq_statistics.jld2",
    "increasing_wind_ep16_k30_beta110_N384_384_256_L10_10_5_hi_freq_statistics.jld2",
    "increasing_wind_ep18_k30_beta110_N384_384_256_L10_10_5_hi_freq_statistics.jld2",

    "increasing_wind_ep12_k30_beta220_N384_384_256_L10_10_5_hi_freq_statistics.jld2",
    "increasing_wind_ep14_k30_beta220_N384_384_256_L10_10_5_hi_freq_statistics.jld2",
    "increasing_wind_ep16_k30_beta220_N384_384_256_L10_10_5_hi_freq_statistics.jld2",
    "increasing_wind_ep18_k30_beta220_N384_384_256_L10_10_5_hi_freq_statistics.jld2",
]
=#

ϵs = [0.14, 0.20, 0.12, 0.14, 0.16, 0.18, 0.12, 0.14, 0.16, 0.18, 0.20]

ϵ_β080 = [0.14, 0.2]
ϵ_β110 = [0.12, 0.14, 0.16, 0.18, 0.20]
ϵ_β220 = [0.12, 0.14, 0.16, 0.18, 0.20]


filenames = [
    "increasing_wind_ep14_k30_beta80_N256_256_192_L10_10_5_hi_freq_statistics.jld2", 
    #"increasing_wind_ep16_k30_beta80_N256_256_192_L10_10_5_hi_freq_statistics.jld2", 
    "increasing_wind_ep20_k30_beta80_N256_256_192_L10_10_5_hi_freq_statistics.jld2",

    "increasing_wind_ep12_k30_beta110_N384_384_256_L10_10_5_hi_freq_statistics.jld2",
    "increasing_wind_ep14_k30_beta110_N512_512_384_L10_10_5_hi_freq_statistics.jld2",
    "increasing_wind_ep16_k30_beta110_N384_384_256_L10_10_5_hi_freq_statistics.jld2",
    "increasing_wind_ep18_k30_beta110_N384_384_256_L10_10_5_hi_freq_statistics.jld2",
    "increasing_wind_ep20_k30_beta110_N256_256_192_L10_10_5_hi_freq_statistics.jld2",

    "increasing_wind_ep12_k30_beta220_N384_384_256_L10_10_5_hi_freq_statistics.jld2", 
    "increasing_wind_ep14_k30_beta220_N384_384_256_L10_10_5_hi_freq_statistics.jld2",
    "increasing_wind_ep16_k30_beta220_N384_384_256_L10_10_5_hi_freq_statistics.jld2",
    "increasing_wind_ep18_k30_beta220_N384_384_256_L10_10_5_hi_freq_statistics.jld2",
    "increasing_wind_ep20_k30_beta220_N384_384_256_L10_10_5_hi_freq_statistics.jld2",

]

#filenames_ep14 = [filenames[[1, 4, 8]]
#filenames_ep20 = [filenames[[1, 4, 8]]

# increasing_wind_ep14_k30_beta110_N256_256_192_L10_10_5_hi_freq_statistics.jld2
# increasing_wind_ep15_k30_beta110_N256_256_192_L10_10_5_hi_freq_statistics.jld2
# increasing_wind_ep15_k30_beta110_N384_384_256_L10_10_5_hi_freq_statistics.jld2
# increasing_wind_ep16_k30_beta110_N256_256_192_L10_10_5_hi_freq_statistics.jld2
# increasing_wind_ep17_k30_beta110_N256_256_192_L10_10_5_hi_freq_statistics.jld2
# increasing_wind_ep17_k30_beta110_N384_384_256_L10_10_5_hi_freq_statistics.jld2
# increasing_wind_ep18_k30_beta110_N256_256_192_L10_10_5_hi_freq_statistics.jld2
# increasing_wind_ep18_k30_beta110_N512_512_384_L10_10_5_hi_freq_statistics.jld2
# increasing_wind_ep19_k30_beta110_N256_256_192_L10_10_5_hi_freq_statistics.jld2
# increasing_wind_ep19_k30_beta110_N384_384_256_L10_10_5_hi_freq_statistics.jld2

function find_transition_times(filenames; wlim=2e-4)
    times = []
    for filename in filenames
        filepath = joinpath(dir, filename)
        stats = compute_timeseries(filepath)
        t = stats[:t]
        wmax = stats[:w_max]

        n = findfirst(wi -> wi >= wlim, wmax)
        isnothing(n) && @show maximum(wmax) filepath
        push!(times, t[n])
    end

    return times
end

fig_w = Figure(resolution=(1200, 400))
ax_w = Axis(fig_w[1, 1], xlabel="t (seconds)", ylabel="max(w) (m s⁻¹)")

for (i, filename) in enumerate(filenames_β220[2:5])
    filepath = joinpath(dir, filename)
    stats = compute_timeseries(filepath)
    t = stats[:t]
    wmax = stats[:w_max]

    lines!(ax_w, t, wmax, label=string("ϵ = ", ϵ_β220[i]))
end

axislegend(ax_w, position=:lt)

fig_t = Figure(resolution=(1200, 400))
ax_t = Axis(fig_t[1, 1], xlabel="ϵ", ylabel="Re★")

filenames_β080 = filenames[1:2]
filenames_β110 = filenames[3:7]
filenames_β220 = filenames[8:end]

t_β080 = find_transition_times(filenames_β080)
t_β110 = find_transition_times(filenames_β110)
t_β220 = find_transition_times(filenames_β220)

ν = 1.05e-6
A_β080 = 8e-6 * sqrt(π / 4ν)
A_β110 = 1.1e-5 * sqrt(π / 4ν)
A_β220 = 2.2e-5 * sqrt(π / 4ν)

Re_β080 = @. A_β080 * sqrt(t_β080^3 / ν)
Re_β110 = @. A_β110 * sqrt(t_β110^3 / ν)
Re_β220 = @. A_β220 * sqrt(t_β220^3 / ν)

#scatter!(ax_t, ϵ_β080, Re_β080, markersize=30, marker=:circle)
scatter!(ax_t, ϵ_β110, Re_β110, markersize=30, marker=:circle, label="τ = 1.1 × 10⁻⁵ \sqrt{t}")
scatter!(ax_t, ϵ_β220, Re_β220, markersize=30, marker=:utriangle, label="τ = 2.2 × 10⁻⁵ \sqrt{t}")

axislegend(ax_t)

#display(fig_w)
display(fig_t)

save("transition_Re.png", fig_t)
save("max_w_transition.png", fig_w)
