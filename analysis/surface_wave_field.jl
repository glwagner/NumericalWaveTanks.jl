using MAT
using CairoMakie
using Statistics

dir = "../data"
filename = "ETAT_R2_allexp.mat"
filepath = joinpath(dir, filename)
vars = matread(filepath)

exp = 1
η = vars["ETA_R2_EXP$exp"]["ETA"]
t = vars["ETA_R2_EXP$exp"]["t"] # ./ 2π
x = vars["ETA_R2_EXP$exp"]["x"]

Y = mean(η, dims=1)[:]
η .-= mean(η, dims=1)

n₀ = 100
n₁ = 220
nn = n₀:n₁
t = t[nn]
η = η[:, nn]
Y = Y[nn]

for sweeps = 1:2
    for i = 1:size(η, 1)
        η′ = view(η, i, :)
        η[i, 2:end-1] = (η′[1:end-2] .+ 2 * η′[2:end-1] .+ η′[3:end]) / 4
    end
end

set_theme!(Theme(fontsize=24, linewidth=3))
fig = Figure(resolution=(1300, 900))
axη = Axis(fig[1, 1], ylabel="Streamwise coordinate (m)", xlabel="Time (s)", xaxisposition=:top)
axs = Axis(fig[2, 1], ylabel="Surface \n displacement (m)", xaxisposition=:top)
axe = Axis(fig[3, 1], ylabel="Steepness \n estimates \n ϵ = a k", xlabel="Time (s)")

hidexdecorations!(axs, grid=false)
hidespines!(axs, :t, :b, :r)
hidespines!(axe, :t, :r)

# t₀_udel = 79.7 / 2π
t₀_udel = 12.4
tᵢ = t[1] - t₀_udel
δ = 1e-3
Δt = t[2] - t[1]
for n = 1:length(t)
    ηn = η[:, n]
    lines!(axη, (ηn[:] ./ δ .+ n) .* Δt .+ tᵢ, x[:], color=(:black, 0.6), linewidth=5)
end

g = 9.81
T = 7.2e-5
k = 2π / 0.03
c = sqrt(g / k + T * k)
phase_speed_colors = Makie.wong_colors(0.6)

t₀ = collect(1.38:0.1:1.9) .+ tᵢ
x₀ = @. 0.065 + c * (t₀ - t₀[1])
lines!(axη, t₀, x₀, color=phase_speed_colors[1], linewidth=10)

k = 2π / 0.05
c = sqrt(g / k + T * k)
t₀ = collect(1.65:0.1:1.95) .+ tᵢ
x₀ = @. 0.005 + c * (t₀ - t₀[1])
lines!(axη, t₀, x₀, color=phase_speed_colors[2], linewidth=10)

troughs = Float64[]
crests = Float64[]
troughs = minimum(η, dims=1)[:]
crests = maximum(η, dims=1)[:]
amplitude = a = @. (crests - troughs) / 2

for sweep = 1:5
    a[2:end-1] .= (a[1:end-2] .+ 2 .* a[2:end-1] .+ a[3:end]) ./ 4 
end

t′ = t .- t₀_udel
lines!(axs, t′, troughs,   linewidth=3, color=(:royalblue, 1.0), label="min η")
lines!(axs, t′, crests,    linewidth=3, color=(:seagreen, 1.0), label="max η")
lines!(axs, t′, amplitude, linewidth=8, color=(:black, 0.3), label="Amplitude estimate")

k1 = 2π / 0.03
k2 = 2π / 0.05

lines!(axe, t′, a * k1,  label="k = 2π / (3 cm)")
lines!(axe, t′, a * k2, label="k = 2π / (5 cm)")

band!(axe, t′, a * k1, a * k2, color=(:black, 0.2))

Legend(fig[2:3, 1], axs, tellwidth=false, halign=:left, margin=(20, 0, 0, 20))
Legend(fig[2:3, 1], axe, tellwidth=false, valign=:center, halign=:center, margin=(0, 0, 0, 20))

ylims!(axs, -2e-3, 2e-3)

rowsize!(fig.layout, 2, Relative(0.2))
rowsize!(fig.layout, 3, Relative(0.2))

xlims!(axη, t′[1] - 1e-2, t′[end] + 1e-1)
xlims!(axs, t′[1] - 1e-2, t′[end] + 1e-1)
xlims!(axe, t′[1] - 1e-2, t′[end] + 1e-1)

display(fig)

save("surface_wave_field_estimates.pdf", fig)
