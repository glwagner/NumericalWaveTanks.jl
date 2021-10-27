using JLD2, Plots, Printf, Oceananigans

# ENV["GKSwstype"] = "100"

name = "veron_and_melville_Nz128_β1.0e-05_waves"

file = jldopen(name * "_averages.jld2")

Nx = file["grid/Nx"]
Ny = file["grid/Ny"]
Nz = file["grid/Nz"]

Lx = file["grid/Lx"]
Ly = file["grid/Ly"]
Lz = file["grid/Lz"]

grid = RegularRectilinearGrid(size = (Nx, Ny, Nz), halo = (3, 3, 3), 
                              x = (0, Lx),
                              y = (0, Ly),
                              z = (-Lz, 0),
                              topology = (Periodic, Bounded, Bounded))

xc, yc, zc = nodes((Center, Center, Center), grid)

iterations = parse.(Int, keys(file["timeseries/t"]))

t = [file["timeseries/t/$iter"] for iter in iterations]

C = zeros(length(t), Nz)

for (i, iter) in enumerate(iterations)
    C[i, :] .= file["timeseries/c/$iter"][1, 1, :]
end

clim = 0.2
clevels = vcat(range(-0.01, clim, length=30), [1.0])

p = contourf(t, zc, C';
             linewidth = 0,
             # aspectratio = :equal,
             xlims = (0, t[end]),
             ylims = (-grid.Lz, 0),
             xlabel = "Time (s)",
             ylabel = "z (m)",
             levels = clevels,
             clims = (0, clim),
             color = :thermal)

display(p)

