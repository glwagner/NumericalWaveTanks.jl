using Oceananigans
using GLMakie
using Printf
using Statistics

include("veron_melville_data.jl")
include("plotting_utilities.jl")

set_theme!(Theme(fontsize=32))

Nx = Ny = 768
Nz = 512
k = 2.1e2
ϵ = 0.18

prefix = "constant_waves_ep140_k30_beta120_N768_768_512_L20_20_10"
dir = "../data"

xy_T_filepath = joinpath(dir, prefix * "_xy_top.jld2")
xy_B_filepath = joinpath(dir, prefix * "_xy_bottom.jld2")
yz_L_filepath = joinpath(dir, prefix * "_yz_left.jld2")
yz_R_filepath = joinpath(dir, prefix * "_yz_right.jld2")
xz_L_filepath = joinpath(dir, prefix * "_xz_left.jld2")
xz_R_filepath = joinpath(dir, prefix * "_xz_right.jld2")

#####
##### Load statistics and averages
#####

statistics_filename = prefix * "_hi_freq_statistics.jld2"
averages_filename   = prefix * "_hi_freq_averages.jld2"
statistics_filepath = joinpath(dir, statistics_filename)
averages_filepath   = joinpath(dir, averages_filename)
stats = compute_timeseries(statistics_filepath)

t_stats = stats[:t]
u_max = stats[:u_max]
w_max = stats[:w_max]
u_min = stats[:u_min]

U = FieldTimeSeries(averages_filepath, "u")
Nz = U.grid.Nz
t = U.times

t_avg = Float64[]
u_avg = Float64[]
Δt = t[2] - t[1]
#u_avg = [U[n][1, 1, Nz] for n in 1:length(U.times)]

for n = 1:length(t)
    if n == 1 || t[n] - t[n-1] >= Δt
        push!(t_avg, t[n])
        push!(u_avg, U[n][1, 1, Nz])
    end
end

Lx = Ly = 0.2
Lz = Ly/2

refinement = 1.5 # controls spacing near surface (higher means finer spaced)
stretching = 8   # controls rate of stretching at bottom
h(k) = (k - 1) / Nz
ζ₀(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))
zf(k) = Lz * (ζ₀(k) * Σ(k) - 1)

file = jldopen(xy_T_filepath)
grid = file["serialized/grid"]
close(file)

#=
grid = RectilinearGrid(CPU(),
                       size = (Nx, Ny, Nz), halo = (3, 3, 3),
                       x = (0, Lx), y = (0, Ly),
                       z = zf,
                       topology = (Periodic, Periodic, Bounded))
=#

u_yz_L_series, t = extract_slices(yz_L_filepath, name="u", dims=1)
c_yz_L_series, t = extract_slices(yz_L_filepath, name="c", dims=1)
u_yz_R_series, t = extract_slices(yz_R_filepath, name="u", dims=1)
c_yz_R_series, t = extract_slices(yz_R_filepath, name="c", dims=1)
       
u_xz_L_series, t = extract_slices(xz_L_filepath, name="u", dims=2)
c_xz_L_series, t = extract_slices(xz_L_filepath, name="c", dims=2)
u_xz_R_series, t = extract_slices(xz_R_filepath, name="u", dims=2)
c_xz_R_series, t = extract_slices(xz_R_filepath, name="c", dims=2)
       
u_xy_T_series, t = extract_slices(xy_T_filepath, name="u", dims=3)
c_xy_T_series, t = extract_slices(xy_T_filepath, name="c", dims=3)
u_xy_B_series, t = extract_slices(xy_B_filepath, name="u", dims=3)
c_xy_B_series, t = extract_slices(xy_B_filepath, name="c", dims=3)

Nt = length(t)
x, y, z = nodes(grid, Face(), Center(), Center())

# Convert to cm
x = 1e2 .* x
y = 1e2 .* y
z = 1e2 .* z
Lx = 1e2 * Lx
Ly = 1e2 * Ly
Lz = 1e2 * Lz

x_xz_L = x_xz_R = repeat(x, 1, Nz)
z_xz_L = z_xz_R = repeat(reshape(z, 1, Nz), Nx, 1)
y_xz_R = 0.995 * Ly * ones(Nx, Nz)
y_xz_L = 0.005 * Ly * ones(Nx, Nz)

y_yz_L = y_yz_R = repeat(y, 1, Nz)
z_yz_L = z_yz_R = repeat(reshape(z, 1, Nz), grid.Ny, 1)
x_yz_R = 0.995 * Lx * ones(Ny, Nz)
x_yz_L = 0.005 * Lx * ones(Ny, Nz)

# Slight displacements to "stitch" the cube together
x_xy_T = x_xy_B = x
y_xy_T = y_xy_B = y
z_xy_T = - 0.001 * Lz * ones(grid.Nx, grid.Ny)
z_xy_B = - 0.995 * Lz * ones(grid.Nx, grid.Ny)

αsim = 0.6
αvm = 0.6
w = 5
azimuth = 5.6
elevation = 0.5
perspectiveness = 1
xlabel = "x (cm)"
ylabel = "y (cm)"
zlabel = "z (cm)"
aspect = :data
xlabeloffset = 80
ylabeloffset = 80
zlabeloffset = 60
u_colorbar_kw = (vertical=true, flipaxis=false, label="x-velocity, u (m s⁻¹)", height=Relative(0.6))
c_colorbar_kw = (vertical=true, flipaxis=true, label="Tracer concentration", height=Relative(0.6))
colormap_u = :solar
colormap_c = :bilbao

fig = Figure(resolution=(1800, 1800))

n = 79
colorrange_c = (0.0, 0.06)
colorrange_u = (0.0, 0.17)
uticks = collect(colorrange_u)
cticks = collect(colorrange_c)

ax_u = fig[1, 2] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness, xlabeloffset, ylabeloffset, zlabeloffset)
ax_c = fig[1, 3] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness, xlabeloffset, ylabeloffset, zlabeloffset)

hidedecorations!(ax_u)
hidedecorations!(ax_c)
hidespines!(ax_u)
hidespines!(ax_c)

u_yz_L = u_yz_L_series[n]
u_xz_L = u_xz_L_series[n]
u_yz_R = u_yz_R_series[n]
u_xz_R = u_xz_R_series[n]
u_xy_T = u_xy_T_series[n]
u_xy_B = u_xy_B_series[n]

c_yz_L = c_yz_L_series[n]
c_xz_L = c_xz_L_series[n]
c_yz_R = c_yz_R_series[n]
c_xz_R = c_xz_R_series[n]
c_xy_T = c_xy_T_series[n]
c_xy_B = c_xy_B_series[n]

pl = surface!(ax_u, x_xz_L, y_xz_L, z_xz_L; color=u_xz_L, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_xz_R, y_xz_R, z_xz_R; color=u_xz_R, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_yz_L, y_yz_L, z_yz_L; color=u_yz_L, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_yz_R, y_yz_R, z_yz_R; color=u_yz_R, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_xy_T, y_xy_T, z_xy_T; color=u_xy_T, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_xy_B, y_xy_B, z_xy_B; color=u_xy_B, colormap=colormap_u, colorrange=colorrange_u)

cp = fig[1, 1] = Colorbar(fig, pl; u_colorbar_kw..., ticks=uticks)

pl = surface!(ax_c, x_xz_L, y_xz_L, z_xz_L; color=c_xz_L, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xz_R, y_xz_R, z_xz_R; color=c_xz_R, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_yz_L, y_yz_L, z_yz_L; color=c_yz_L, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_yz_R, y_yz_R, z_yz_R; color=c_yz_R, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xy_T, y_xy_T, z_xy_T; color=c_xy_T, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xy_B, y_xy_B, z_xy_B; color=c_xy_B, colormap=colormap_c, colorrange=colorrange_c)

cp = fig[1, 4] = Colorbar(fig, pl; c_colorbar_kw..., ticks=cticks)

# Label(fig[1, 9], @sprintf("t = %.1f seconds", t[n]), tellheight=false)

n = 88
colorrange_c = (0.0, 0.03)
colorrange_u = (0.0, 0.12)
uticks = collect(colorrange_u)
cticks = collect(colorrange_c)

ax_u = fig[2, 2] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness, xlabeloffset, ylabeloffset, zlabeloffset)
ax_c = fig[2, 3] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness, xlabeloffset, ylabeloffset, zlabeloffset)

hidedecorations!(ax_u)
hidedecorations!(ax_c)
hidespines!(ax_u)
hidespines!(ax_c)

u_yz_L = u_yz_L_series[n]
u_xz_L = u_xz_L_series[n]
u_yz_R = u_yz_R_series[n]
u_xz_R = u_xz_R_series[n]
u_xy_T = u_xy_T_series[n]
u_xy_B = u_xy_B_series[n]

c_yz_L = c_yz_L_series[n]
c_xz_L = c_xz_L_series[n]
c_yz_R = c_yz_R_series[n]
c_xz_R = c_xz_R_series[n]
c_xy_T = c_xy_T_series[n]
c_xy_B = c_xy_B_series[n]

pl = surface!(ax_u, x_xz_L, y_xz_L, z_xz_L; color=u_xz_L, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_xz_R, y_xz_R, z_xz_R; color=u_xz_R, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_yz_L, y_yz_L, z_yz_L; color=u_yz_L, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_yz_R, y_yz_R, z_yz_R; color=u_yz_R, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_xy_T, y_xy_T, z_xy_T; color=u_xy_T, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_xy_B, y_xy_B, z_xy_B; color=u_xy_B, colormap=colormap_u, colorrange=colorrange_u)

cp = fig[2, 1] = Colorbar(fig, pl; u_colorbar_kw..., ticks=uticks)

pl = surface!(ax_c, x_xz_L, y_xz_L, z_xz_L; color=c_xz_L, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xz_R, y_xz_R, z_xz_R; color=c_xz_R, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_yz_L, y_yz_L, z_yz_L; color=c_yz_L, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_yz_R, y_yz_R, z_yz_R; color=c_yz_R, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xy_T, y_xy_T, z_xy_T; color=c_xy_T, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xy_B, y_xy_B, z_xy_B; color=c_xy_B, colormap=colormap_c, colorrange=colorrange_c)

cp = fig[2, 4] = Colorbar(fig, pl; c_colorbar_kw..., ticks=cticks)

n = 120
colorrange_c = (0.0, 0.008)
colorrange_u = (0.0, 0.12)
uticks = collect(colorrange_u)
cticks = collect(colorrange_c)

ax_u = fig[3, 2] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness, xlabeloffset, ylabeloffset, zlabeloffset)
ax_c = fig[3, 3] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness, xlabeloffset, ylabeloffset, zlabeloffset)

# hidedecorations!(ax_u)
# hidedecorations!(ax_c)
hidespines!(ax_u)
hidespines!(ax_c)

u_yz_L = u_yz_L_series[n]
u_xz_L = u_xz_L_series[n]
u_yz_R = u_yz_R_series[n]
u_xz_R = u_xz_R_series[n]
u_xy_T = u_xy_T_series[n]
u_xy_B = u_xy_B_series[n]

c_yz_L = c_yz_L_series[n]
c_xz_L = c_xz_L_series[n]
c_yz_R = c_yz_R_series[n]
c_xz_R = c_xz_R_series[n]
c_xy_T = c_xy_T_series[n]
c_xy_B = c_xy_B_series[n]

pl = surface!(ax_u, x_xz_L, y_xz_L, z_xz_L; color=u_xz_L, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_xz_R, y_xz_R, z_xz_R; color=u_xz_R, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_yz_L, y_yz_L, z_yz_L; color=u_yz_L, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_yz_R, y_yz_R, z_yz_R; color=u_yz_R, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_xy_T, y_xy_T, z_xy_T; color=u_xy_T, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_xy_B, y_xy_B, z_xy_B; color=u_xy_B, colormap=colormap_u, colorrange=colorrange_u)

cp = fig[3, 1] = Colorbar(fig, pl; u_colorbar_kw..., ticks=uticks)

pl = surface!(ax_c, x_xz_L, y_xz_L, z_xz_L; color=c_xz_L, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xz_R, y_xz_R, z_xz_R; color=c_xz_R, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_yz_L, y_yz_L, z_yz_L; color=c_yz_L, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_yz_R, y_yz_R, z_yz_R; color=c_yz_R, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xy_T, y_xy_T, z_xy_T; color=c_xy_T, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xy_B, y_xy_B, z_xy_B; color=c_xy_B, colormap=colormap_c, colorrange=colorrange_c)

cp = fig[3, 4] = Colorbar(fig, pl; c_colorbar_kw..., ticks=cticks)

colgap!(fig.layout, 2, -200)
rowgap!(fig.layout, 1, -200)
rowgap!(fig.layout, 2, -200)

# Label(fig[2, 9], @sprintf("t = %.1f seconds", t[n]), tellheight=false)

display(fig)

save("turbulence_visualization.png", fig)

# record(fig, "langmuir_turbulence.mp4", 1:Nt; framerate=12) do nn
#     @info "Drawing frame $nn of $Nt..."
#     n[] = nn
# end

