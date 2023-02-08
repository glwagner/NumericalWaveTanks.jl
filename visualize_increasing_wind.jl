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
fig = Figure(resolution=(2400, 1600))
ax_u = fig[1:w, 1:4] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness)
ax_c = fig[1:w, 5:8] = Axis3(fig; aspect, xlabel, ylabel, zlabel, azimuth, elevation, perspectiveness)

Nt = length(t)
#slider = Slider(fig[w+2, :], range=1:Nt, horizontal=true, startvalue=1)
n = Observable(1) #slider.value

u_yz_L = @lift u_yz_L_series[$n]
u_xz_L = @lift u_xz_L_series[$n]
u_yz_R = @lift u_yz_R_series[$n]
u_xz_R = @lift u_xz_R_series[$n]
u_xy_T = @lift u_xy_T_series[$n]
u_xy_B = @lift u_xy_B_series[$n]

c_yz_L = @lift c_yz_L_series[$n]
c_xz_L = @lift c_xz_L_series[$n]
c_yz_R = @lift c_yz_R_series[$n]
c_xz_R = @lift c_xz_R_series[$n]
c_xy_T = @lift c_xy_T_series[$n]
c_xy_B = @lift c_xy_B_series[$n]

ax_s = Axis(fig[1, 2:7],
            xlabel="Time (seconds)",
            ylabel="Surface velocity \n (m s⁻¹)")

xlims!(ax_s, 0.0, 30.0)

u_max_past = deepcopy(u_max)
u_min_past = deepcopy(u_min)
u_avg_past = deepcopy(u_avg)

u_max_plot = @lift begin
    u_max_past .= u_max
    u_max_past[t_stats .> t[$n]] .= NaN
    u_max_past
end

u_min_plot = @lift begin
    u_min_past .= u_min
    u_min_past[t_stats .> t[$n]] .= NaN
    u_min_past
end

u_avg_plot = @lift begin
    u_avg_past .= u_avg
    u_avg_past[t_avg .> t[$n]] .= NaN
    u_avg_past
end

lines!(ax_s, t_stats, u_max_plot, linewidth=6, color=(:red,    αsim), label="max(u), simulation")
lines!(ax_s, t_stats, u_min_plot, linewidth=6, color=(:orange, αsim), label="min(u), simulation")
lines!(ax_s, t_avg,   u_avg_plot, linewidth=4, color=(:blue,   αsim), label="mean(u), simulation")

lines!(ax_s, t_stats, u_max, linewidth=6, color=(:red,    αsim/4))
lines!(ax_s, t_stats, u_min, linewidth=6, color=(:orange, αsim/4))
lines!(ax_s, t_avg,   u_avg, linewidth=4, color=(:blue,   αsim/4))

time_text = @lift @sprintf("t = %.2f seconds", t[$n])
text!(ax_s, 1.0, 0.11, text=time_text, textsize=36)

time_marker = @lift t[$n]
vlines!(ax_s, time_marker, linewidth=6, color=(:black, 0.6))

t₀_vm = 2.0
t_vm_surf = veron_melville_data[:t_surf] .- t₀_vm
u_vm_surf = veron_melville_data[:u_surf] ./ 100

t_vm_avg_surf = veron_melville_data[:t_avg_surf] .- t₀_vm
u_vm_avg_surf = veron_melville_data[:u_avg_surf] ./ 100

t_vm_jet = veron_melville_data[:t_jet] .- t₀_vm
u_vm_jet = veron_melville_data[:u_jet] ./ 100

t_vm_wake = veron_melville_data[:t_wake] .- t₀_vm
u_vm_wake = veron_melville_data[:u_wake] ./ 100

scatter!(ax_s, t_vm_surf, u_vm_surf, marker=:utriangle, markersize=20, color=(:blue, αvm),
         label="Surface velocity, Veron and Melville (2001)")

scatter!(ax_s, t_vm_avg_surf, u_vm_avg_surf, marker=:rect, markersize=20, color=(:purple, αvm),
         label="Average surface velocity, Veron and Melville (2001)")

scatter!(ax_s, t_vm_jet, u_vm_jet, markersize=10, color=(:indigo, αvm),
         label="Jet velocity, Veron and Melville (2001)")

scatter!(ax_s, t_vm_wake, u_vm_wake, marker=:cross, markersize=20, color=(:cyan, 1.0),
         label="Wake velocity, Veron and Melville (2001)")

# Legend(fig[0, 1], ax_u, tellwidth=false)

colormap_u = :balance
colormap_c = :solar
colorrange_c = (0.0, 0.015)
colorrange_u = (-0.12, 0.12)

#=
colorrange_u = @lift begin
    umax = 1e2 * maximum(abs, u_xy_T_series[$n])
    ulim = max(1e-2, 0.8 * umax)
    (-ulim/10, ulim)
end
=#

#=
Δc = 0.05
colorrange_c = @lift begin
    cmax = maximum(c_xy_T_series[$n])
    cmin = minimum(c_xz_L_series[$n])
    Δc = cmax - cmin
    #(cmin + Δc/2, cmax)
    (0.0, cmax / 2)
end
=#

pl = surface!(ax_u, x_xz_L, y_xz_L, z_xz_L; color=u_xz_L, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_xz_R, y_xz_R, z_xz_R; color=u_xz_R, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_yz_L, y_yz_L, z_yz_L; color=u_yz_L, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_yz_R, y_yz_R, z_yz_R; color=u_yz_R, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_xy_T, y_xy_T, z_xy_T; color=u_xy_T, colormap=colormap_u, colorrange=colorrange_u)
     surface!(ax_u, x_xy_B, y_xy_B, z_xy_B; color=u_xy_B, colormap=colormap_u, colorrange=colorrange_u)

cp = fig[w+1, 2:3] = Colorbar(fig, pl, vertical=false, flipaxis = false, label="x-velocity, u (m s⁻¹)")

pl = surface!(ax_c, x_xz_L, y_xz_L, z_xz_L; color=c_xz_L, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xz_R, y_xz_R, z_xz_R; color=c_xz_R, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_yz_L, y_yz_L, z_yz_L; color=c_yz_L, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_yz_R, y_yz_R, z_yz_R; color=c_yz_R, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xy_T, y_xy_T, z_xy_T; color=c_xy_T, colormap=colormap_c, colorrange=colorrange_c)
     surface!(ax_c, x_xy_B, y_xy_B, z_xy_B; color=c_xy_B, colormap=colormap_c, colorrange=colorrange_c)

cp = fig[w+1, 6:7] = Colorbar(fig, pl, vertical=false, flipaxis=false, label="Tracer concentration")

#lab = Label(fig[0, :], title, textsize=24)

display(fig)

record(fig, "langmuir_turbulence.mp4", 1:Nt; framerate=12) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end

