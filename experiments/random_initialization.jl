using FourierFlows
using Oceananigans
using GLMakie
using Statistics

Nx = 128
Ny = 128
Nz = 64

Lx = 0.1
Ly = 0.1
Lz = 0.05

spectral_grid = ThreeDGrid(nx=Nx, ny=Ny, nz=Nz; Lx, Ly, Lz)
k = sqrt.(spectral_grid.Krsq)

FT = Float64
θu = randn(Complex{FT}, size(spectral_grid.Krsq))
θv = randn(Complex{FT}, size(spectral_grid.Krsq))
θw = randn(Complex{FT}, size(spectral_grid.Krsq))

û = @. θu / k
v̂ = @. θv / k
ŵ = @. θw / k

@show size(k)

û[1, 1, 1] = 0
v̂[1, 1, 1] = 0
ŵ[1, 1, 1] = 0

u₀  = irfft(û, spectral_grid.nx)
v₀  = irfft(v̂, spectral_grid.nx)
w₀′ = irfft(ŵ, spectral_grid.nx)

w₀ = zeros(Nx, Ny, Nz+1)
w₀[:, :, 2:Nz] .= w₀′[:, :, 2:Nz]

grid = RectilinearGrid(size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(-Lz, 0))
model = NonhydrostaticModel(; grid)

Ξ(x, y, z) = randn()
#set!(model, u=u₀, v=v₀, w=w₀)
set!(model, u=Ξ, v=Ξ, w=Ξ)

u, v, w = model.velocities
e = Field((u^2 + v^2 + w^2) / 2)
compute!(e)
@show mean(e)

ui = Array(interior(u))

ũ = rfft(ui, 1)
ẽ = @. abs2(ũ)
E = mean(ẽ, dims=(2, 3))[:, 1, 1]

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, E)
display(fig)

