using GLMakie
using Oceananigans
using MAT
using Printf

lif_filename = "../data/TRANSVERSE_STAT_RAMP2_LIF_final.mat"
lif_data = matread(lif_filename)["STAT_R2"]

# Load LIF data as "concentration"
c = lif_data["LIFa"]
c = permutedims(c, (2, 3, 1)) # puts time in last dimension

# Load time (convert from 2D array to 1D vector)
t = lif_data["time"][:]
Nt = length(t)

x = lif_data["X_transverse_m"][:]
z = lif_data["Z_transverse_m"][:] .- 0.11

fig = Figure(resolution=(2700, 900))
ax = Axis(fig[2, 1], xlabel="Along-wind direction (m)", ylabel="z (m)")
slider = Slider(fig[3, 1], range=1:Nt, startvalue=127)
n = slider.value

title = @lift @sprintf("LIF data frame %d at t = %.2f seconds", $n, t[$n])
Label(fig[1, 1], title, tellwidth=false)

j1 = 1200
j2 = 1650
cn = @lift rotr90(view(c, :, :, $n))[:, j1:j2]
heatmap!(ax, x, z[j1:j2], cn, colorrange=(0, 1500))

display(fig)

record(fig, "compare_lif_simulation.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end

#=
n = 130
fig = Figure(resolution=(1600, 1600))

ax = Axis(fig[2, 1])

title = "LIF data frame $n at t = " * prettytime(t[n])
Label(fig[1, 1], title, tellwidth=false)

slider = Slider(fig[3, 1], range=1:100, startvalue=50)
ϵ = slider.value

cn = rotr90(view(c, :, :, n))[:, 1200:1700]
colorrange = @lift begin
    lower = 0
    upper = $ϵ / 100 * maximum(abs, cn)
    (lower, upper)
end

heatmap!(ax, cn; colorrange) #(0, clim))

display(fig)

=#
