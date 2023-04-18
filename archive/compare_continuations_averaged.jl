using Oceananigans
using JLD2
using GLMakie
using Printf
using Statistics

no_waves_filename = "continued_increasing_wind_256_256_256_k2.1e+02_ep0.0e+00_averages.jld2"
med_waves_filename = "continued_increasing_wind_256_256_256_k2.1e+02_ep1.0e-01_averages.jld2"
str_waves_filename = "continued_increasing_wind_256_256_256_k2.1e+02_ep3.0e-01_averages.jld2"

Nx = Ny = Nz = 256
Lx = Ly = 0.2
Lz = Ly/2

refinement = 1.5 # controls spacing near surface (higher means finer spaced)
stretching = 8   # controls rate of stretching at bottom
h(k) = (k - 1) / Nz
ζ₀(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

grid = RectilinearGrid(CPU(),
                       size = (Nx, Ny, Nz), halo = (3, 3, 3),
                       x = (0, Lx), y = (0, Ly),
                       z = k -> Lz * (ζ₀(k) * Σ(k) - 1), # (-Lz, 0)
                       topology = (Periodic, Periodic, Bounded))

function hovmoller(filepath; name, dims=(1, 2))
    file = jldopen(filepath)
    iterations = parse.(Int, keys(file["timeseries/t"]))
    slices = [dropdims(file["timeseries/$name/$i"]; dims) for i in iterations]
    hov = hcat(slices...)
    times = [file["timeseries/t/$i"] for i in iterations]
    close(file)
    return permutedims(hov, (2, 1)), times
end

c_no_waves, t = hovmoller(no_waves_filename, name="c")
c_med_waves, t = hovmoller(med_waves_filename, name="c")
c_str_waves, t = hovmoller(str_waves_filename, name="c")
   
x, y, z = nodes((Face, Center, Center), grid)

fig = Figure(resolution=(1400, 800))

ax1 = Axis(fig[1, 1])
heatmap!(ax1, z, t, c_no_waves)

ax2 = Axis(fig[2, 1])
heatmap!(ax2, z, t, c_no_waves .- c_str_waves)

display(fig)

#=
# Convert to cm
x = 1e2 .* x
y = 1e2 .* y
z = 1e2 .* z
Lx = 1e2 * Lx
Ly = 1e2 * Ly
Lz = 1e2 * Lz

x_xz = repeat(x, 1, Nz)
y_xz = 0.995 * Ly * ones(Nx, Nz)
z_xz = repeat(reshape(z, 1, Nz), Nx, 1)

x_yz = 0.995 * Lx * ones(Ny, Nz)
y_yz = repeat(y, 1, Nz)
z_yz = repeat(reshape(z, 1, Nz), grid.Ny, 1)

# Slight displacements to "stitch" the cube together
x_xy = x
y_xy = y
z_xy = - 0.001 * Lz * ones(grid.Nx, grid.Ny)

azimuth = 3.7
elevation = 0.4
perspectiveness = 1
xlabel = "x (cm)"
ylabel = "y (cm)"
zlabel = "z (cm)"
aspect = :data
fig = Figure(resolution=(1400, 800))
w = 5
ax_u = fig[1:w, 1:4] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness)
ax_c = fig[1:w, 5:8] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness)

Nt = length(c_yz_series)
slider = Slider(fig[w+2, :], range=1:Nt, horizontal=true, startvalue=1)
n = slider.value

u_yz = @lift 1e2 .* u_yz_series[$n]
u_xz = @lift 1e2 .* u_xz_series[$n]
u_xy = @lift 1e2 .* u_xy_series[$n]

c_yz = @lift c_yz_series[$n]
c_xz = @lift c_xz_series[$n]
c_xy = @lift c_xy_series[$n]

colormap_u = :oslo
colorrange_u = @lift begin
    umax = 1e2 * maximum(abs, u_xy_series[$n])
    ulim = max(1e-2, 0.8 * umax)
    (-ulim/10, ulim)
end

Δc = 0.02
colormap_c = :solar
colorrange_c = @lift begin
    cmax = maximum(c_xy_series[$n])
    cmin = minimum(c_xz_series[$n])
    Δc = cmax - cmin
    (cmin + Δc/2, cmax)
end

pl = surface!(ax_u, x_xz, y_xz, z_xz; color=u_xz, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_yz, y_yz, z_yz; color=u_yz, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_xy, y_xy, z_xy; color=u_xy, colormap=colormap_u, colorrange=colorrange_u)

cp = fig[w+1, 2:3] = Colorbar(fig, pl, vertical=false, flipaxis = false, label="x-velocity, u (cm s⁻¹)")

pl = surface!(ax_c, x_xz, y_xz, z_xz; color=c_xz, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_yz, y_yz, z_yz; color=c_yz, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xy, y_xy, z_xy; color=c_xy, colormap=colormap_c, colorrange=colorrange_c)

cp = fig[w+1, 6:7] = Colorbar(fig, pl, vertical=false, flipaxis=false, label="Tracer concentration")

title = @lift @sprintf("Laboratory scale currents beneath increasing wind and surface waves after %.1f seconds", t[$n])
lab = Label(fig[0, :], title, textsize=24)

display(fig)

#record(fig, "langmuir_turbulence.mp4", 1:Nt; framerate=25) do nn
#    @info "Drawing frame $nn of $Nt..."
#    slider.value[] = nn
#end

=#
