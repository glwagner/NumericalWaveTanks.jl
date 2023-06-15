using Oceananigans
using GLMakie
using Printf

#filepath = "../data/sudden_waves_ep300_k30_beta120_N512_512_256_L20_20_10_yz_left.jld2"
filepath = "../data/constant_waves_ep140_k30_beta120_N512_512_384_L20_20_10_yz_left.jld2"

ut = FieldTimeSeries(filepath, "u")
ct = FieldTimeSeries(filepath, "c")
t = ut.times
Nt = length(t)

fig = Figure()
axu = Axis(fig[2, 1])
axc = Axis(fig[2, 2])
slider = Slider(fig[3, 1:2], range=1:Nt, startvalue=1)
n = slider.value

title = @lift string("t = ", prettytime(t[$n]))
Label(fig[1, 1:2], title)

u = @lift interior(ut[$n], 1, :, :)
c = @lift interior(ct[$n], 1, :, :)

heatmap!(axu, u)
heatmap!(axc, c)

display(fig)

