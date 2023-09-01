using CairoMakie
using JLD2
using MAT
using Oceananigans
using Oceananigans.Operators: Δz

set_theme!(Theme(fontsize=24))

include("veron_melville_data.jl")
include("plotting_utilities.jl")

dir = "../data"
case = "constant_waves_ic000000_050000_ep110_k30_alpha120_N768_768_512_L10_10_5"
#case = "constant_waves_ic000000_050000_ep100_k30_alpha120_N768_768_512_L10_10_5"
statistics_filename = case * "_statistics.jld2"
averages_filename   = case * "_averages.jld2"

statistics_filepath = joinpath(dir, statistics_filename)
averages_filepath   = joinpath(dir, averages_filename)

new_filename = "../data/fig4b.mat"
ramp = "2"
t₀_udel = 79.7
lif_threshold = 1e-4
z_threshold = 1e-4
lif_filename = "../data/fig4b.mat"

#####
##### Load LIF data
#####

lif_data = matread(lif_filename)["fig4b"]
t_lif = lif_data["time"][:] .- t₀_udel
c_lif = permutedims(lif_data["LIF"], (2, 1))
z_lif = -lif_data["z"][:] .+ 0.034
N_lif = length(t_lif)

#####
##### Analyze LIF data
#####

# Shift
c_lif_min = minimum(c_lif)
c_lif = @. c_lif - c_lif_min

# Normalize
C_lif = sum(c_lif, dims=2)
c_lif = c_lif ./ C_lif[1]

# Clip
zz = repeat(z_lif', length(t_lif), 1)
c_lif[c_lif .< lif_threshold] .= 0
c_lif[zz .> z_threshold] .= 0

# Compute integrated LIF and centroid for equispaced grid
C_lif = sum(c_lif, dims=2)
Z_lif = [sum(z_lif .* c_lif[n, :]) / C_lif[n] for n = 1:N_lif]

#####
##### Load simulation data
#####

U = FieldTimeSeries(averages_filepath, "u")
Nz = U.grid.Nz
u_avg = [U[n][1, 1, Nz] for n in 1:length(U.times)]
t_avg = U.times

stats = compute_timeseries(statistics_filepath)
t_stats = stats[:t]
u_max = stats[:u_max]
w_max = stats[:w_max]
u_min = stats[:u_min]

ct = FieldTimeSeries(averages_filepath, "c")
grid = ct.grid
c_sim = interior(ct, 1, 1, :, :)
c_sim = permutedims(c_sim, (2, 1))
z_sim = znodes(grid, Center())
t_sim = ct.times
Nt = length(t_sim)

const c = Center()
Δzs = zspacings(grid, c, c, c) 
C_sim = sum(c_sim .* reshape(Δzs, 1, grid.Nz), dims=2)
Z_sim = zeros(size(c_sim, 1))
Z₉₉_sim = zeros(size(c_sim, 1))

for n = 1:length(Z_sim)
    Z_sim[n] = sum(z_sim .* Δzs .* c_sim[n, :]) / C_sim[n]

    C⁺ = 0
    C⁻ = 0
    for k = Nz:-1:1
        C⁻ = C⁺
        C⁺ += Δzs[k] * c_sim[n, k]
        if C⁺ > 0.99 * C_sim[n]
            @show k
            Z₉₉_sim[n] = z_sim[k]
            break
        end
    end
end

c_sim = c_sim ./ C_sim
c_sim = @. max(c_sim, 0, c_sim)

# Figure
set_theme!(Theme(fontsize=32, linewidth=4))
t₀ = 15.0
t₁ = 30.0
z₀ = -5
z₁ = 0.02
fig = Figure(resolution=(1600, 1200))
colormap = :bilbao

yticks = -5:1:0
xticks = -0.1:0.02:0.0
# ax_int = Axis(fig[1, 1]; xlabel="Simulation time (seconds)", ylabel="Normalized ∫c dz")
ax_sim = Axis(fig[1, 1]; xlabel="Time (s)", ylabel="z (cm)", yticks)
ax_lif = Axis(fig[2, 1]; xlabel="Time (s)", ylabel="z (cm)", yticks)
ax_cen = Axis(fig[3, 1]; xlabel="Time (s)", ylabel="Z(t) = ∫z c dz (cm)")

lines!(ax_cen, t_sim, 1e2 .* Z_sim, label="Simulation")
lines!(ax_cen, t_lif, 1e2 .* Z_lif, label="LIF")
axislegend(ax_cen, position=:rt)

heatmap!(ax_lif, t_lif, 1e2 .* z_lif, log10.(c_lif); colorrange=(-4, -2.5), colormap)

xlims!(ax_cen, t₀, t₁)
ylims!(ax_cen, -3, 0.3)
xlims!(ax_lif, t₀, t₁)
ylims!(ax_lif, z₀, z₁)

# Clip negative values
heatmap!(ax_sim, t_sim, 1e2 .* z_sim, log10.(c_sim); colormap, colorrange=(-5, 2.5))
lines!(ax_sim, t_sim, 1e2 .* Z₉₉_sim, color=:dodgerblue, label="Simulated Z₉₉(t)", linewidth=6)

xlims!(ax_sim, t₀, t₁)
ylims!(ax_sim, z₀, z₁)

colors = Makie.wong_colors()
lines!(ax_lif, t_sim, 1e2 .* Z₉₉_sim, color=:dodgerblue, label="Simulated Z₉₉(t)", linewidth=6)
#band!(ax_lif,  t_sim, 1e2 .* Z₉₉_sim, zeros(length(t_sim)), color=(:lightblue1, 0.2), linewidth=6)

colors = Makie.wong_colors(0.6)
vlines!(ax_sim, 16, color=colors[1], label="t = 16 s (ripple inception)")
vlines!(ax_sim, 18, color=colors[2], label="t = 18 s (self-sharpening)")
vlines!(ax_sim, 20, color=colors[3], label="t = 20 s (transition to Langmuir turbulence)")

text!(ax_sim, 0.01, 0.04, text="(a) Simulation", space=:relative, align=(:left, :bottom))
text!(ax_lif, 0.01, 0.04, text="(b) Laboratory", space=:relative, align=(:left, :bottom))
text!(ax_cen, 0.01, 0.04, text="(c) Centroid comparison", space=:relative, align=(:left, :bottom))

vlines!(ax_lif, 16, color=colors[1]) #, label="Ripple inception")
vlines!(ax_lif, 18, color=colors[2]) #, label="Self-sharpening")
vlines!(ax_lif, 20, color=colors[3]) #, label="Three-dimensionalization")

axislegend(ax_sim, position=:rt, framecolor=:white, bgcolor=(:white, 0.8))
axislegend(ax_lif, position=:rt, framecolor=:white, bgcolor=(:white, 0.8))

#=
dir = "../data"
case = "changed_waves_epf0_ic000000_050000_ep110_k30_alpha120_N768_768_512_L10_10_5"
averages_filename   = case * "_averages.jld2"

ct = FieldTimeSeries(averages_filepath, "c")
grid = ct.grid
c_sim = interior(ct, 1, 1, :, :)
c_sim = permutedims(c_sim, (2, 1))
z_sim = znodes(grid, Center())
t_sim = ct.times
Nt = length(t_sim)

const c = Center()
Δzs = zspacings(grid, c, c, c) 
C_sim = sum(c_sim .* reshape(Δzs, 1, grid.Nz), dims=2)
Z_sim = zeros(size(c_sim, 1))
Z₉₉_sim = zeros(size(c_sim, 1))

for n = 1:length(Z_sim)
    Z_sim[n] = sum(z_sim .* Δzs .* c_sim[n, :]) / C_sim[n]

    C⁺ = 0
    C⁻ = 0
    for k = Nz:-1:1
        C⁻ = C⁺
        C⁺ += Δzs[k] * c_sim[n, k]
        if C⁺ > 0.99 * C_sim[n]
            @show k
            Z₉₉_sim[n] = z_sim[k]
            break
        end
    end
end

lines!(ax_sim, t_sim, 1e2 .* Z₉₉_sim, color=:seagreen, label="Simulated Z₉₉(t) from changed waves", linewidth=6)
=#

display(fig)

save("hovmoller.pdf", fig)
