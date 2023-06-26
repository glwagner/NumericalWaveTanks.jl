using JLD2
using GLMakie
using Oceananigans
using Statistics
using Printf
using FourierFlows

set_theme!(Theme(fontsize=32, linewidth=3, markersize=20))

Ny = 768
Ly = 0.1
small_grid = OneDGrid(nx=Ny, Lx=Ly)

Ny = 3072
Ly = 0.4
big_grid = OneDGrid(nx=Ny, Lx=Ly)

grids = [small_grid, big_grid]

resolutions = ["N768_512_L10_5",
               "N3072_256_L40_2"]

steepnesses = 0.02:0.01:0.3

fig = Figure(resolution=(1700, 500))
axr = Axis(fig[1, 1], xlabel="Wave steepness, ϵ", ylabelpadding=20, ylabel="Growth rate (s⁻¹)")
axs = Axis(fig[1, 2], xlabel="Wave steepness, ϵ", ylabelpadding=20, ylabel="Growth time-scale (s)")
axl = Axis(fig[1, 3], xlabel="Wave steepness, ϵ", ylabelpadding=20, ylabel="Most unstable \n wavelength (cm)")

xlims!(axr, -0.01, 0.31)
xlims!(axs, -0.01, 0.31)
xlims!(axl, -0.01, 0.31)
ylims!(axs, 0, 2.4)

colors = Makie.wong_colors(0.8)
markers = [:circle, :utriangle]
markersizes = [20, 15]
labels = ["L = 10 cm", "L = 40 cm"]

for n = 1:2
    grid = grids[n]
    resolution = resolutions[n]
    kr = grid.kr
    growth_rates = Float64[]
    wavenumbers = Float64[]

    for ϵ in steepnesses
        prefix = @sprintf("linearly_unstable_mode_t0160_ep%02d_%s", 100ϵ, resolution)
        filename = prefix * ".jld2"

        file = jldopen(filename)
        u = file["u"]
        σ = file["growth_rate"]
        close(file)

        u0 = u[1, 4:end-3, end-3] # surface
        û = abs.(rfft(u0))
        umax, j = findmax(û)

        @show kr[j]

        push!(wavenumbers, kr[j])
        push!(growth_rates, σ)
    end

    if n == 1
        scaling = steepnesses[3:end] .- steepnesses[end]
        lines!(axr, steepnesses[3:end], 27.5 .* scaling .+ growth_rates[end], color=(:black, 0.3), linewidth=10)
    end

    marker = markers[n]
    color = colors[n]
    markersize = markersizes[n]
    label = labels[n]
    scatter!(axr, steepnesses, growth_rates; marker, color, markersize, label)
    scatter!(axs, steepnesses, 1 ./ growth_rates; marker, color, markersize, label)
    scatter!(axl, steepnesses, 1e2 * 2π ./ wavenumbers; marker, color, markersize, label)
end

axislegend(axs)
text!(axr, 0.25, 0.08, text="σ ~ 27.5 ϵ", space=:relative, color=:gray, fontsize=32)
display(fig)
save("linear_stability_summary.pdf", fig)

