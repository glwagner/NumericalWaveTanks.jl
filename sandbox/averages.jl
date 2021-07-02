using JLD2, Plots, Printf, Oceananigans

ENV["GKSwstype"] = "100"

#name = "veron_and_melville_Nz128_β1.0e-04"
name = "veron_and_melville_Nz128_β1.0e-05_waves"

yz_file = jldopen(name * "_yz.jld2")
xz_file = jldopen(name * "_xz.jld2")

Nx = yz_file["grid/Nx"]
Ny = yz_file["grid/Ny"]
Nz = yz_file["grid/Nz"]

Lx = yz_file["grid/Lx"]
Ly = yz_file["grid/Ly"]
Lz = yz_file["grid/Lz"]

grid = RegularRectilinearGrid(size = (Nx, Ny, Nz), halo = (3, 3, 3), 
                              x = (0, Lx),
                              y = (0, Ly),
                              z = (-Lz, 0),
                              topology = (Periodic, Bounded, Bounded))

xw, yw, zw = nodes((Center, Center, Face), grid)
xu, yu, zu = nodes((Face, Center, Center), grid)
xc, yc, zc = nodes((Center, Center, Center), grid)

iterations = parse.(Int, keys(yz_file["timeseries/t"]))

t = [xz_file["timeseries/t/$iter"] for iter in iterations]
C = [mean(xz_file["timeseries/c/$iter"][:, 1, :], dims=1)[:, 1] for iter in iterations]

p = contourf(t, zc, C,
             linewidth = 0,
             # aspectratio = :equal,
             xlims = (0, t[end]),
             ylims = (-grid.Lz, 0),
             color = :thermal)

display(p)
                       #levels = ulevels,
                       #clims = (-ulim, ulim))

