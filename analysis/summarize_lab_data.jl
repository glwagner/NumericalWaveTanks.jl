using GLMakie #CairoMakie
using JLD2
using MAT
using Oceananigans
using Printf
using Statistics

include("veron_melville_data.jl")
include("plotting_utilities.jl")

dir = "../data"
exp = 2
exp_str = "R$exp"
ramp = 2
t₀ = 79.5

######
###### Load wave data
######

filename = "ETAT_R2_allexp.mat"
filepath = joinpath(dir, filename)
vars = matread(filepath)

η = vars["ETA_R2_EXP$exp"]["ETA"]
t_wave = vars["ETA_R2_EXP$exp"]["t"]
x_wave = vars["ETA_R2_EXP$exp"]["x"]
Nt_wave = length(t_wave)

η .-= mean(η, dims=1)

η = η[:, :]
t_wave = t_wave[:] .- t₀

for sweeps = 1:2
    for i = 1:size(η, 1)
        η′ = view(η, i, :)
        η[i, 2:end-1] = (η′[1:end-2] .+ 2 * η′[2:end-1] .+ η′[3:end]) / 4
    end
end

troughs = Float64[]
crests = Float64[]
troughs = minimum(η, dims=1)[:]
crests = maximum(η, dims=1)[:]
amplitude = a = @. (crests - troughs) / 2

for sweep = 1:5
    a[2:end-1] .= (a[1:end-2] .+ 2 .* a[2:end-1] .+ a[3:end]) ./ 4 
end

######
###### Load surface velocity data
######

surface_velocity_filename = joinpath(dir, "every_surface_velocity.mat")
surface_velocity_data = matread(surface_velocity_filename)
U_surface = surface_velocity_data["BIN"][exp_str]["U"][:]
t_surface = surface_velocity_data["BIN"][exp_str]["time"][:]
t_surface = t_surface .- t₀

#####
##### Load LIF data as "concentration"
#####

lif_filename = "../data/TRANSVERSE_STAT_RAMP2_LIF_final.mat"
lif_data = matread(lif_filename)["STAT_R2"]

c_lif = lif_data["LIFa"]
c_lif = permutedims(c_lif, (2, 3, 1)) # puts time in last dimension

# Load time (convert from 2D array to 1D vector)
t_lif = lif_data["time"][:] .- t₀
x_lif = lif_data["X_transverse_m"][:]
z_lif = lif_data["Z_transverse_m"][:] .- 0.108
Nt = length(t_lif)

x_lif .-= minimum(x_lif)

# Convert to cm
x_lif .*= 1e2
z_lif .*= 1e2

# Figure
set_theme!(Theme(fontsize=28))
fig = Figure(resolution=(1400, 1400))

k1 = 2π / 0.03
k2 = 2π / 0.05

xticks = 0:5:30
ax_e = Axis(fig[1, 1],
            ylabel = "Steepness \n estimates \n ϵ = a k",
            xlabel = "Time (s)",
            yscale = log10,
            yticks = ([0.03, 0.1, 0.3], ["0.03", "0.1", "0.3"]),
            xticks = xticks,
            xaxisposition = :top)

text!(ax_e, 0.01, 0.98, text="(a)", space=:relative, align=(:left, :top))

lines!(ax_e, t_wave, a * k1, linewidth=3, label="k = 2π / (3 cm)")
lines!(ax_e, t_wave, a * k2, linewidth=3, label="k = 2π / (5 cm)")
 band!(ax_e, t_wave, a * k1, a * k2, color=(:black, 0.2))
axislegend(ax_e, position=:lb)

ax_u = Axis(fig[2, 1],
            xticks = xticks,
            xlabel = "Time (s)",
            ylabel = "Measured \n along-wind \n velocity (cm s⁻¹)")

hidespines!(ax_e, :b)
hidespines!(ax_u, :t)

t₁ = -2
t₂ = 30
xlims!(ax_e, t₁, t₂)
ylims!(ax_e, 0.03, 0.5)
xlims!(ax_u, t₁, t₂)
ylims!(ax_u, 0, 25)

scatter!(ax_u, t_surface, 1e2 .* U_surface, marker=:circle, markersize=20, color=:black,
         label="Measured surface velocity (cm s⁻¹)")

text!(ax_u, 0.01, 0.98, text="(b)", space=:relative, align=(:left, :top))

A = 1e-2
t★ = 0:0.1:20.0
U★ = A .* t★
lines!(ax_u, t★, U★ .* 1e2, color=(:black, 0.3), linewidth=15,
       label="U = 10⁻² t (cm s⁻¹)")

text!(ax_u, 7, 9, text="U ~ t (cm s⁻¹)", color=(:black, 0.6), rotation=0.25)

#axislegend(ax_u, position=:lt)

colorrange = (100, 1500)
colormap = :bilbao
j1 = 1350
j2 = 1660

t1 = 18
t2 = 19.3
t3 = 22.3

n1 = searchsortedfirst(t_lif, t1)
n2 = searchsortedfirst(t_lif, t2)
n3 = searchsortedfirst(t_lif, t3)

t1_lif = t_lif[n1]
t2_lif = t_lif[n2]
t3_lif = t_lif[n3]

c1 = rotr90(view(c_lif, :, :, n1))[:, j1:j2]
c2 = rotr90(view(c_lif, :, :, n2))[:, j1:j2]
c3 = rotr90(view(c_lif, :, :, n3))[:, j1:j2]

ymin1 = 0.45
ymin2 = 0.2
ymin3 = 0.2

ymax1 = 0.67
ymax2 = 0.5
ymax3 = 0.45

xlabel1 = t1_lif
xlabel2 = t2_lif
xlabel3 = t3_lif

ylabel1 = 6.5
ylabel2 = 2.0
ylabel3 = 2.0

Lx = maximum(x_lif) - minimum(x_lif)
Lz = maximum(z_lif[j1:j2]) - minimum(z_lif[j1:j2])
aspect = Lx / Lz

#slider = Slider(fig[3, 1], range=0:0.1:30, startvalue=18)
#tn = slider.value

colors = (:royalblue1, :crimson, :darkolivegreen) #Makie.wong_colors()[1:end]

ax_c1 = Axis(fig[3, 1]; xlabel="Cross-wind direction (cm)", ylabel="z (cm)", aspect)
heatmap!(ax_c1, x_lif, z_lif[j1:j2], c1; colorrange, colormap=:bilbao)
label  = @sprintf("(c) t = %.2f seconds", t1_lif)
text!(ax_c1, 0.1, z_lif[j1] + 0.1, text=label, color=colors[1], align=(:left, :bottom))

#vlines!(ax_u, t1_lif, ymin=ymin1, ymax=ymax1, color=(colors[1], 0.5), linewidth=6)
arrows!(ax_u, [t1_lif], [ymin1], [0], [(ymax1-ymin1)], color=(colors[1], 0.5), linewidth=6)
text!(ax_u, xlabel1, ylabel1, text=@sprintf("%.2f", t1_lif), color=colors[1], align=(:right, :center))

tn = Observable(19)
ax_c2 = Axis(fig[4, 1]; xlabel="Across-wind direction (cm)", ylabel="z (cm)", aspect)
heatmap!(ax_c2, x_lif, z_lif[j1:j2], c2; colorrange, colormap=:bilbao)
label  = @sprintf("(d) t = %.2f seconds", t2_lif)
text!(ax_c2, 0.1, z_lif[j1] + 0.1, text=label, color=colors[2], align=(:left, :bottom))

vlines!(ax_u, t2_lif, ymin=ymin2, ymax=ymax2, color=(colors[2], 0.6), linewidth=6)
text!(ax_u, xlabel2, ylabel2, text=@sprintf("%.2f", t2_lif), color=colors[2], align=(:right, :center))

ax_c3 = Axis(fig[5, 1]; xlabel="Across-wind direction (cm)", ylabel="z (cm)", aspect)
heatmap!(ax_c3, x_lif, z_lif[j1:j2], c3; colorrange, colormap=:bilbao)
label  = @sprintf("(e) t = %.2f seconds", t3_lif)
text!(ax_c3, 7.0, z_lif[j1] + 0.1, text=label, color=colors[3], align=(:left, :bottom))

vlines!(ax_u, t3_lif, ymin=ymin3, ymax=ymax3, color=(colors[3], 0.6), linewidth=6)
text!(ax_u, xlabel3, ylabel3, text=@sprintf("%.2f", t3_lif), color=colors[3], align=(:left, :center))

hidexdecorations!(ax_c1)
hidexdecorations!(ax_c2)

rowgap!(fig.layout, 1, 50)
rowsize!(fig.layout, 3, Relative(0.2))
rowsize!(fig.layout, 4, Relative(0.2))
rowsize!(fig.layout, 5, Relative(0.2))

display(fig)

#save("lab_data_summary.pdf", fig)

