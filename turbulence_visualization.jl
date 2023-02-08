using Oceananigans
using GLMakie
using Printf
using Statistics

include("veron_melville_data.jl")
include("plotting_utilities.jl")

set_theme!(Theme(fontsize=36))

Nx = Ny = 512
Nz = 384
k = 2.1e2
ϵ = 0.18

prefix = "increasing_wind_ep135_k30_beta120_N512_512_384_L10_10_5"
#prefix = "increasing_wind_ep14_k30_beta110_N384_384_256_L10_10_5"
dir = "data"

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

Lx = Ly = 0.1
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

x, y, z = nodes((Face, Center, Center), grid)

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
xlabeloffset = 60
ylabeloffset = 60
zlabeloffset = 60

colormap_u = :balance
colormap_c = :solar
colorrange_c = (0.0, 0.01)
colorrange_u = (-0.12, 0.12)

fig = Figure(resolution=(2400, 1600))

ax_u = fig[1, 1:4] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness, xlabeloffset, ylabeloffset, zlabeloffset)
ax_c = fig[1, 5:8] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness, xlabeloffset, ylabeloffset, zlabeloffset)

Nt = length(t)

# slider = Slider(fig[4, :], range=1:Nt, horizontal=true, startvalue=1)
# n = slider.value

n = 91

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

pl = surface!(ax_c, x_xz_L, y_xz_L, z_xz_L; color=c_xz_L, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xz_R, y_xz_R, z_xz_R; color=c_xz_R, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_yz_L, y_yz_L, z_yz_L; color=c_yz_L, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_yz_R, y_yz_R, z_yz_R; color=c_yz_R, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xy_T, y_xy_T, z_xy_T; color=c_xy_T, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xy_B, y_xy_B, z_xy_B; color=c_xy_B, colormap=colormap_c, colorrange=colorrange_c)

Label(fig[1, 9], @sprintf("t = %.1f seconds", t[n]), tellheight=false)

ax_u = fig[2, 1:4] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness, xlabeloffset, ylabeloffset, zlabeloffset)
ax_c = fig[2, 5:8] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness, xlabeloffset, ylabeloffset, zlabeloffset)

n = 137

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

cp = fig[3, 2:3] = Colorbar(fig, pl, vertical=false, flipaxis = false, label="x-velocity, u (m s⁻¹)", width=Relative(0.6),
                            ticks = [-0.1, 0.0, 0.1])

pl = surface!(ax_c, x_xz_L, y_xz_L, z_xz_L; color=c_xz_L, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xz_R, y_xz_R, z_xz_R; color=c_xz_R, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_yz_L, y_yz_L, z_yz_L; color=c_yz_L, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_yz_R, y_yz_R, z_yz_R; color=c_yz_R, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xy_T, y_xy_T, z_xy_T; color=c_xy_T, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xy_B, y_xy_B, z_xy_B; color=c_xy_B, colormap=colormap_c, colorrange=colorrange_c)

cp = fig[3, 6:7] = Colorbar(fig, pl, vertical=false, flipaxis=false, label="Tracer concentration", width=Relative(0.6),
                            ticks = [0.0, 0.005, 0.01])

colgap!(fig.layout, 4, -1000)
colgap!(fig.layout, 8, -400)
#rowgap!(fig.layout, 2, -100)
rowgap!(fig.layout, 1, -150)
rowgap!(fig.layout, 2, +60)

Label(fig[2, 9], @sprintf("t = %.1f seconds", t[n]), tellheight=false)

display(fig)

save("turbulence_visualization.png", fig)

# record(fig, "langmuir_turbulence.mp4", 1:Nt; framerate=12) do nn
#     @info "Drawing frame $nn of $Nt..."
#     n[] = nn
# end

