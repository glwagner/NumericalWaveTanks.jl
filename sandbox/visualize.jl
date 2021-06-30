using JLD2, Plots, Printf, Oceananigans

ENV["GKSwstype"] = "100"

#name = "veron_and_melville_Nz128_β1.0e-04"
name = "veron_and_melville_Nz128_β1.0e-06_waves"

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

iterations = parse.(Int, keys(yz_file["timeseries/t"]))

anim = @animate for (i, iter) in enumerate(iterations)
    wyz = yz_file["timeseries/w/$iter"][1, :, :]
    uyz = yz_file["timeseries/u/$iter"][1, :, :]
    uxz = xz_file["timeseries/u/$iter"][:, 1, :]

    t = yz_file["timeseries/t/$iter"]

    wmax = maximum(abs, wyz)
    wlim = wmax / 2
    wlevels = range(-wlim, stop=wlim, length=30)
    wlim < wmax && (wlevels = vcat([-wmax], wlevels, [wmax]))

    umax = maximum(abs, uxz)
    ulim = umax / 2
    ulevels = range(-ulim, stop=ulim, length=30)
    ulim < umax && (ulevels = vcat([-umax], ulevels, [umax]))

    w_title = @sprintf("w(y, z, t) (m s⁻¹) at t = %s ", prettytime(t))
    u_title = @sprintf("u(y, z, t) (m s⁻¹) at t = %s ", prettytime(t))

    @show w_title

    uxzplot = contourf(xu, zu, uxz';
                       linewidth = 0,
                       aspectratio = :equal,
                       xlims = (0, grid.Lx),
                       ylims = (-grid.Lz, 0),
                       color = :balance,
                       levels = ulevels,
                       clims = (-ulim, ulim))

    wyzplot = contourf(yw, zw, wyz';
                       linewidth = 0,
                       aspectratio = :equal,
                       xlims = (0, grid.Ly),
                       ylims = (-grid.Lz, 0),
                       color = :balance,
                       levels = wlevels,
                       clims = (-wlim, wlim))

    uyzplot = contourf(yu, zu, uyz';
                       linewidth = 0,
                       aspectratio = :equal,
                       xlims = (0, grid.Ly),
                       ylims = (-grid.Lz, 0),
                       color = :balance,
                       levels = ulevels,
                       clims = (-ulim, ulim))

    layout = @layout [
        a
        Plots.grid(1, 2)
    ]

    plot(uxzplot, wyzplot, uyzplot,
         layout = layout,
         size = (1200, 1200),
         title = [u_title w_title u_title])
end

close(yz_file)
close(xz_file)

gif(anim, name * ".gif", fps=8)
