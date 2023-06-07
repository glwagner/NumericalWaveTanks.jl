using GLMakie
using Oceananigans
using MAT

lif_filename = "TRANSVERSE_STAT_RAMP1_LIF_final.mat"
ramp_1_data = matread(lif_filename)["STAT_R1"]

# Load LIF data as "concentration"
c = ramp_1_data["LIFa"]
c = permutedims(c, (2, 3, 1)) # puts time in last dimension

# Load time (convert from 2D array to 1D vector)
t = ramp_1_data["time"][:]
Nt = length(t)

x = ramp_1_data["X_transverse_m"][:]
z = ramp_1_data["Z_transverse_m"][:] .- 0.11

fig = Figure(resolution=(1200, 800))
ax = Axis(fig[2, 1])
slider = Slider(fig[3, 1], range=1:Nt, startvalue=127)
n = slider.value

title = @lift string("LIF data frame ", $n, " at t = ", prettytime(t[$n]))
Label(fig[1, 1], title, tellwidth=false)

j1 = 1200
j2 = 1650
cn = @lift rotr90(view(c, :, :, $n))[:, j1:j2]
heatmap!(ax, x, z[j1:j2], cn, colorrange=(0, 1500))

display(fig)

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
