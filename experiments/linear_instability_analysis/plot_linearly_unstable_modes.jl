using JLD2
using CairoMakie #GLMakie
using Oceananigans
using Statistics
using Printf
using FFTW

Ny = 768
Nz = 512
Ly = 0.1
Lz = 0.05

refinement = 1.5 # controls spacing near surface (higher means finer spaced)
stretching = 8   # controls rate of stretching at bottom
h(k) = (k - 1) / Nz
ζ₀(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

grid = RectilinearGrid(size = (Ny, Nz),
                       halo = (3, 3),
                       y = (0, Ly),
                       z = k -> Lz * (ζ₀(k) * Σ(k) - 1), # (-Lz, 0)
                       topology = (Flat, Periodic, Bounded))

y = 1e2 .* ynodes(grid, Center())[:]
zf = 1e3 .* znodes(grid, Face())[:]
zc = 1e3 .* znodes(grid, Center())[:]

set_theme!(Theme(fontsize=48, linewidth=5))

#for ϵ = 0.06:0.01:0.12
    ϵ = 0.1
    prefix = @sprintf("linearly_unstable_mode_t0160_ep%02d_N768_512_L10_5", 100ϵ)
    filename = prefix * ".jld2"

    file = jldopen(filename)
    u = file["u"]
    v = file["v"]
    w = file["w"]
    # grid = file["grid"]
    σ = file["growth_rate"]
    close(file)

    @show size(u)

    @info @sprintf("σ: %.2f, σ⁻¹: %.2f", σ, 1/σ)

    fig = Figure(resolution=(2800, 420))
    title = @sprintf("Most unstable eigenmode, ϵ = %.2f, σ⁻¹: %.2f seconds", ϵ, 1/σ)
    xticks = [0, 2, 4, 6, 8]
    yticks = [-12, -8, -4,  0]
    axv = Axis(fig[1, 1]; aspect=10/1.2, xticks, yticks, xlabel="y (cm)", ylabel="z (mm)")
    axz = Axis(fig[1, 2]; yticks, ylabel="z (mm)", xlabel="rms(u′)", yaxisposition=:right)

    #hidexdecorations!(axv)

    u₀ = maximum(abs, u)
    hmv = heatmap!(axv, y, zc, v[1, 4:end-3, 4:end-3] ./ u₀, colorrange=(-0.1, 0.1), colormap=:balance)
    #Colorbar(fig[1, 1], hmv, label="v′ / max |u′|", flipaxis=false)
    text!(axv, 0.005, 0.04, text="(a) v′(y, z)", space=:relative, align=(:left, :bottom))

    ui = u[1, 4:end-3, 4:end-3]
    vi = v[1, 4:end-3, 4:end-3]
    wi = w[1, 4:end-3, 4:end-3]
    ũ = rfft(ui, 1)
    ṽ = rfft(vi, 1)
    w̃ = rfft(wi, 1)
    umax, ju = findmax(abs.(ũ[:, 1]))
    vmax, jv = findmax(abs.(ṽ[:, 1]))
    wmax, jw = findmax(abs.(w̃[:, 1]))

    @show ju jv jw
    
    û = real.(ũ[ju, :])
    v̂ = real.(ṽ[jv, :])
    ŵ = real.(w̃[jw, :])

    U = sqrt.(mean(u.^2, dims=2))
    V = sqrt.(mean(v.^2, dims=2))
    W = sqrt.(mean(w.^2, dims=2))

    u₀ = maximum(abs, U)
    lines!(axz, U[1, 1, 4:end-3] ./ u₀, zc, label="u′")
    lines!(axz, V[1, 1, 4:end-3] ./ u₀, zc, label="v′")
    lines!(axz, W[1, 1, 4:end-3] ./ u₀, zf, label="w′")
    text!(axz, 0.1, 0.04, text="(b)", space=:relative, align=(:left, :bottom))

    #=
    ℓ = 13 * 2π / grid.Ly
    Yf = YFaceField(grid)
    Yc = CenterField(grid)
    set!(Yf, (x, y, z) -> y)
    set!(Yc, (x, y, z) -> y)
    U = Field(Integral(u′ * cos(ℓ * Yc), dims=2))
    V = Field(Integral(v′ * sin(ℓ * Yf), dims=2))
    W = Field(Integral(w′ * cos(ℓ * Yc), dims=2))
    compute!(U)
    compute!(V)
    compute!(W)

    u₀ = maximum(abs, U)

    lines!(axz, interior(U, 1, 1, :) ./ u₀, zc, label="û")
    lines!(axz, interior(V, 1, 1, :) ./ u₀, zc, label="v̂")
    lines!(axz, interior(W, 1, 1, :) ./ u₀, zf, label="ŵ")
    =#

    # u₀ = maximum(abs, û)
    # lines!(axz, û ./ u₀, zc, label="û")
    # lines!(axz, v̂ ./ u₀, zc, label="v̂")
    # lines!(axz, ŵ ./ u₀, zf, label="ŵ")

    axislegend(axz, position=:rb, framevisible=false)

    colsize!(fig.layout, 2, Relative(0.1))
    for ax in (axv, axz)
        ylims!(ax, -12, 0)
    end

    figname = prefix * ".pdf"
    save(figname, fig)

    display(fig)

#    sleep(1.0)
#end

