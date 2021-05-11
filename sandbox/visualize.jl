using JLD2, Plots, Printf, Oceananigans

filename = "veron_and_melville_Nz128_yz.jld2"
yz_file = jldopen(filename)

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

iterations = parse.(Int, keys(yz_file["timeseries/t"]))

anim = @animate for (i, iter) in enumerate(iterations)
    wyz = yz_file["timeseries/w/$iter"][1, :, :]
    t = yz_file["timeseries/t/$iter"]

    wmax = maximum(abs, wyz)
    wlim = wmax / 2
    wlevels = range(-wlim, stop=wlim, length=30)
    wlim < wmax && (wlevels = vcat([-wmax], wlevels, [wmax]))

    wyz_title = @sprintf("w(y, z, t) (m s⁻¹) at t = %s ", prettytime(t))
    @show wyz_title

    contourf(yw, zw, wyz';
             linewidth = 0,
             aspectratio = :equal,
             xlims = (0, grid.Ly),
             ylims = (-grid.Lz, 0),
             color = :balance,
             levels = wlevels,
             clims = (-wlim, wlim),
             title = wyz_title)
end

close(yz_file)

gif(anim, filename[1:end-5] * ".gif", fps=8)
