#####
##### Langmuir-cell diagnostics from the surface x-y slices (xy_surface.jld2, seed 1 runs written
##### with animation=true): the two-dimensional spectrum of surface vertical velocity, split into
##### streamwise-elongated modes (|k_x| < |k_y|/3, the signature of Langmuir rolls) and the rest,
##### and the x and y integral scales of w. Usage:
#####   julia --project=. analysis/anti_stokes/langmuir_diagnostics.jl runs=<dir1>,<dir2>,... [ages=-2,0,1,2,3]
#####

using FFTW
include("common.jl")

args = parse_key_value_args(ARGS)
dirs = split(args["runs"], ',')
ages = parse.(Float64, split(getarg(args, "ages", "-2,0,1,2,3"), ','))

function corr_length(A, Δ; dim)
    # integral scale from the autocorrelation along dim (periodic), averaged over the other dim
    B = A .- mean(A)
    F = fft(B, dim)
    R = real.(ifft(abs2.(F), dim))
    R = dropdims(mean(R; dims = dim == 1 ? 2 : 1); dims = dim == 1 ? 2 : 1)
    R ./= R[1]
    n = findfirst(<=(0), R)
    n = isnothing(n) ? length(R) ÷ 2 : n
    return Δ * (0.5 + sum(R[2:n-1]; init=0.0))
end

for d in dirs
    run = load_run(d; fields=("U",))
    τ, tp = τ₀(run), t_peak(run)
    w_ts = FieldTimeSeries(joinpath(d, "xy_surface.jld2"), "w")
    t = collect(Float64, w_ts.times)
    grid = w_ts.grid
    Lx, Ly = run.meta["Lx"], run.meta["Ly"]
    Nx, Ny = size(interior(w_ts[1]))[1:2]
    kx = 2π / Lx .* fftfreq(Nx, Nx); ky = 2π / Ly .* fftfreq(Ny, Ny)
    println("\n", run.meta["member"], " (", run.meta["level"], ", seed ", run.meta["seed"], ")")
    println("   age    w_rms(mm/s)  elongated fraction  L_x(m)  L_y(m)   L_x/L_y   peak k_y (m⁻¹) of elongated modes")
    for a in ages
        n = nearest_index(t, tp + a * τ)
        w = Float64.(Array(interior(w_ts[n]))[:, :, 1])
        w .-= mean(w)
        F = abs2.(fft(w))
        F[1, 1] = 0
        total = sum(F)
        elong = [abs(kx[i]) < abs(ky[j]) / 3 && ky[j] != 0 for i in 1:Nx, j in 1:Ny]
        frac = sum(F[elong]) / total
        # spanwise spectrum of the elongated modes
        Sy = [sum(F[elong[:, j], j]) for j in 1:Ny]
        jpk = argmax(Sy[2:Ny÷2]) + 1
        Lxw, Lyw = corr_length(w, Lx / Nx; dim=1), corr_length(w, Ly / Ny; dim=2)
        @printf("  %+4.1f   %8.2f      %6.3f          %6.3f  %6.3f   %6.2f    %6.1f (λ_y = %.2f m)\n",
                a, 1e3 * sqrt(mean(abs2, w)), frac, Lxw, Lyw, Lxw / Lyw, abs(ky[jpk]), 2π / abs(ky[jpk]))
    end
end
