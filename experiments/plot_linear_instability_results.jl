using JLD2
using Printf

inception_times = Dict()
growth_rates = Dict()

ϵs = 0.05:0.05:0.3
for ϵ in ϵs
    filepath = @sprintf("linear_instability_analysis_ep%02d_3.jld2", 100ϵ)
    file = jldopen(filepath)
    inception_times[ϵ] = file["inception_times"]
    growth_rates[ϵ] = file["time_independent_growth_rates"]
    close(file)
end

fig = Figure()
ax = Axis(fig[1, 1], yscale=identity, xlabel="Inception time (s)", ylabel="Doubling time (s)")
#ax = Axis(fig[1, 1], xlabel="Inception time (s)", ylabel="Growth rate (s⁻¹)")

for ϵ in ϵs
    σ = growth_rates[ϵ]
    t = inception_times[ϵ]
    scatterlines!(ax, t, log(2) ./ σ)
    #scatterlines!(ax, t, σ)
end

display(fig)

