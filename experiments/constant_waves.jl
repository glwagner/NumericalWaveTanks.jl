using CUDA
using Random
using Statistics
using SpecialFunctions
using OrderedCollections
using JLD2
using MPI
using Oceananigans
using Oceananigans: fields
using Oceananigans.Forcings: regularize_forcing
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.DistributedComputations: Distributed, Partition, mpi_rank
using Oceananigans.Units
using Printf

# Helper: select architecture from a (Px, Py, Pz) partition. Initializes MPI on
# first call when distribution is requested. Pz is currently always 1 (no z split)
# because the FFT pressure solver requires the z-direction to be local.
function select_architecture(Px, Py, Pz)
    if Px * Py * Pz > 1
        MPI.Initialized() || MPI.Init()
        return Distributed(GPU(); partition = Partition(Px, Py, Pz))
    else
        return GPU()
    end
end

is_root() = !MPI.Initialized() || MPI.Comm_rank(MPI.COMM_WORLD) == 0

κ_rhodamine = 1e-7 # find a reference for this

# Constant Stokes shear utility
struct ConstantStokesShear{T}
    a :: T
    k :: T
    ω :: T
end

ConstantStokesShear(a, k; g=9.81, γ=7.2e-5) = ConstantStokesShear{Float64}(a, k, sqrt(g * k + γ * k^3))
@inline (sh::ConstantStokesShear)(z, t) = 2 * sh.a^2 * sh.k^2 * sh.ω * exp(2 * sh.k * z) 

# Stokes streaming
@inline ν_∂z²_uˢ(x, y, z, t, p) = - 4 * p.ν * p.∂z_uˢ.a^2 * p.∂z_uˢ.k^3 * p.∂z_uˢ.ω * exp(2 * p.∂z_uˢ.k * z)

function build_numerical_wave_tank(arch;
                                   # Cross-stream direction
                                   Ny = 256,
                                   Ly = 0.2,
                                   # Along-stream direction
                                   Nx = Ny,
                                   Lx = Ly,
                                   # Vertical direction
                                   Nz = round(Int, Ny/2),
                                   Lz = Ly / 2,
                                   ϵ = 0.0,
                                   k = 2π/0.03,
                                   ν = 1.05e-6,
                                   κ = κ_rhodamine,
                                   α = 1.2e-5,
                                   t₀ = 0.0,
                                   W′ = 1e-4,
                                   stop_time = 22.0,
                                   save_interval = 0.2,
                                   save_interval_3d = 0.5,
                                   overwrite_existing = true,
                                   name = "constant_waves")

    refinement = 1.5 # controls spacing near surface (higher means finer spaced)
    stretching = 8   # controls rate of stretching at bottom
    h(k) = (k - 1) / Nz
    ζ₀(k) = 1 + (h(k) - 1) / refinement
    Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

    @info "Building a grid..." 
    grid = RectilinearGrid(arch,
                           size = (Nx, Ny, Nz),
                           halo = (3, 3, 3),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = k -> Lz * (ζ₀(k) * Σ(k) - 1), # (-Lz, 0)
                           topology = (Periodic, Periodic, Bounded))

    @show grid

    #####
    ##### Surface stress
    #####

    @inline τʷ(x, y, t, p) = - p.α * sqrt(t)
    u_top_bc = FluxBoundaryCondition(τʷ, parameters = (; α))
        
    u_bcs = FieldBoundaryConditions(top = u_top_bc)
    boundary_conditions = (; u = u_bcs)

    #####
    ##### Stokes streaming term with a forcing function
    #####
    
    a = ϵ / k
    ∂z_uˢ = ConstantStokesShear(a, k)
    u_forcing = Forcing(ν_∂z²_uˢ, parameters=(; ∂z_uˢ, ν))

    #####
    ##### The model
    #####

    vitd = VerticallyImplicitTimeDiscretization()
  
    model = NonhydrostaticModel(grid; boundary_conditions,
                                advection = Centered(order=2),
                                timestepper = :RungeKutta3,
                                tracers = :c,
                                closure = ScalarDiffusivity(vitd; ν, κ),
                                stokes_drift = UniformStokesDrift(; ∂z_uˢ),
                                forcing = (; u = u_forcing))

    ω = ∂z_uˢ.ω

    @info """

        Wave parameters | Values
        =============== | ======
                      a | $a
                      k | $k
                 2π / k | $(2π / k)
                      ϵ | $ϵ
    """

    #####
    ##### Initial condition: mean flow + perturbations
    #####

    # Mean flow
    A = α * sqrt(π / 4ν)
    U₀ = A * t₀
    h = √(2 * ν * t₀)

    function Uᵢ(x, y, z)
        δ = z / h
        Ξ = 10W′ * randn()
        return Ξ + U₀ * ((1 + δ^2) * erfc(-δ / √2) + δ * √(2/π) * exp(-δ^2 / 2))
    end

    wᵢ(x, y, z) = 10W′ * randn()

    set!(model, u=Uᵢ, v=wᵢ, w=wᵢ)

    # Optionally add precomputed eigenmode perturbations on top of the random noise.
    # If the eigenmode file is missing, fall back to random noise alone — useful for
    # tests at coarse resolutions where no eigenmode has been computed.
    eigen_filename = @sprintf("linearly_unstable_mode_t0%02d_ep%02d_N%d_%d_L%d_%d.jld2",
                              10t₀, 100ϵ, Ny, Nz, 100Ly, 100Lz)
    eigen_filepath = joinpath("linear_instability_analysis", eigen_filename)

    if isfile(eigen_filepath)
        @info "Loading eigenmode IC from $eigen_filepath"
        file = jldopen(eigen_filepath)
        û = file["u"]
        v̂ = file["v"]
        ŵ = file["w"]
        close(file)

        ArrayType = arch isa CPU ? Array : CuArray
        u′ = ArrayType(û)
        v′ = ArrayType(v̂)
        w′ = ArrayType(ŵ)

        W = maximum(abs, ŵ)
        u′ .*= W′ / W
        v′ .*= W′ / W
        w′ .*= W′ / W

        u, v, w = model.velocities
        parent(u) .+= u′
        parent(v) .+= v′
        parent(w) .+= w′
    else
        @info "Eigenmode file $eigen_filepath not found; using random IC only"
    end

    model.clock.time = t₀

    c = model.tracers.c
    view(interior(c), :, :, grid.Nz) .= 1
    
    #####
    ##### Set up simulation
    #####

    @info "Revvving up a simulation..."
    simulation = Simulation(model; Δt=1e-4, stop_time)

    #Δ = min(minimum(parent(grid.Δzᵃᵃᶜ)), grid.Δxᶜᵃᵃ)
    Δ = grid.Δxᶜᵃᵃ
    @show max_Δt = 0.1 * Δ^2 / ν
    wizard = TimeStepWizard(; cfl=1.0, max_Δt)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))

    wall_clock = Ref(time_ns())

    # Closure over wall_clock
    function progress(sim)

        umax = maximum(abs, sim.model.velocities.u)
        vmax = maximum(abs, sim.model.velocities.v)
        wmax = maximum(abs, sim.model.velocities.w)

        t = time(sim)
        h = √(ν * t)
        Re = umax * h / ν
        elapsed = 1e-9 * (time_ns() - wall_clock[])

        @info @sprintf("Time: %s, iter: %d, Δt: %s, wall time: %s, max|U|: (%.2e, %.2e, %.2e)  m s⁻¹",
                       prettytime(t),
                       iteration(sim),
                       prettytime(sim.Δt),
                       prettytime(elapsed),
                       umax, vmax, wmax)

        wall_clock[] = time_ns()

        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    #####
    ##### Set up output
    #####

    Nx, Ny, Nz = size(model.grid)

    file_prefix = @sprintf("%s_ic%06d_ep%03d_k%d_alpha%d_N%d_%d_%d_L%d_%d_%d",
                           name, 1e6 * W′, 1000ϵ, 1000 * 2π/k, 1e7 * α,
                           Nx, Ny, Nz,
                           100 * model.grid.Lx,
                           100 * model.grid.Ly,
                           100 * model.grid.Lz)

    nobackup_dir = "."
    dir = joinpath(nobackup_dir, file_prefix)

    # In distributed mode each rank writes its own files into the shared
    # output directory; suffix all filenames/prefixes with the rank index.
    # `analysis/combine_dns_snapshots.jl` stitches the rank shards back
    # into a single global file for downstream LES restart.
    rank_suffix = MPI.Initialized() ?
                  @sprintf("_rank%03d", MPI.Comm_rank(MPI.COMM_WORLD)) :
                  ""

    is_root() && @info "Saving data to $file_prefix"

    outputs = merge(model.velocities, model.tracers)

    u, v, w = model.velocities
    C = Field(Average(model.tracers.c, dims=(1, 2)))
    U = Field(Average(u, dims=(1, 2)))

    simulation.output_writers[:avg] = JLD2Writer(model, (c=C, u=U); dir, overwrite_existing,
                                                       schedule = TimeInterval(save_interval),
                                                       filename = file_prefix * "_averages" * rank_suffix)

    simulation.output_writers[:fast_avg] = JLD2Writer(model, (c=C, u=U); dir, overwrite_existing,
                                                            schedule = TimeInterval(0.02),
                                                            filename = file_prefix * "_hi_freq_averages" * rank_suffix)

    Nz = grid.Nz

    statistics = (u_max = model -> maximum(abs, view(interior(model.velocities.u), :, :, Nz)), 
                  u_min = model -> minimum(abs, view(interior(model.velocities.u), :, :, Nz)),
                  v_max = model -> maximum(abs, view(interior(model.velocities.v), :, :, Nz)),
                  w_max = model -> maximum(abs, model.velocities.w))

    simulation.output_writers[:stats] = JLD2Writer(model, statistics; dir, overwrite_existing,
                                                         schedule = TimeInterval(save_interval),
                                                         filename = file_prefix * "_statistics" * rank_suffix)

    simulation.output_writers[:hi_freq_stats] = JLD2Writer(model, statistics; dir, overwrite_existing,
                                                                 schedule = TimeInterval(0.02),
                                                                 filename = file_prefix * "_hi_freq_statistics" * rank_suffix)

    simulation.output_writers[:yz_left] = JLD2Writer(model, outputs; dir, overwrite_existing,
                                                           schedule = TimeInterval(save_interval),
                                                           filename = file_prefix * "_yz_left" * rank_suffix,
                                                           indices = (1, :, :))

    simulation.output_writers[:xz_left] = JLD2Writer(model, outputs; dir, overwrite_existing,
                                                           schedule = TimeInterval(save_interval),
                                                           filename = file_prefix * "_xz_left" * rank_suffix,
                                                           indices = (:, 1, :))

    simulation.output_writers[:xy_bottom] = JLD2Writer(model, outputs; dir, overwrite_existing,
                                                             schedule = TimeInterval(save_interval),
                                                             filename = file_prefix * "_xy_bottom" * rank_suffix,
                                                             indices = (:, :, 1))

    simulation.output_writers[:yz_right] = JLD2Writer(model, outputs; dir, overwrite_existing,
                                                            schedule = TimeInterval(save_interval),
                                                            filename = file_prefix * "_yz_right" * rank_suffix,
                                                            indices = (grid.Nx, :, :))

    simulation.output_writers[:xz_right] = JLD2Writer(model, outputs; dir, overwrite_existing,
                                                            schedule = TimeInterval(save_interval),
                                                            filename = file_prefix * "_xz_right" * rank_suffix,
                                                            indices = (:, grid.Ny, :))

    simulation.output_writers[:xy_top] = JLD2Writer(model, outputs; dir, overwrite_existing,
                                                          schedule = TimeInterval(save_interval),
                                                          filename = file_prefix * "_xy_top" * rank_suffix,
                                                          indices = (:, :, grid.Nz))

    # 3D state snapshots for using as LES initial conditions.
    # Saved as Float32 without halos to keep file sizes manageable.
    simulation.output_writers[:fields_3d] = JLD2Writer(model, outputs; dir, overwrite_existing,
                                                       schedule = TimeInterval(save_interval_3d),
                                                       filename = file_prefix * "_3d_fields" * rank_suffix,
                                                       array_type = Array{Float32},
                                                       with_halos = false)

    # Full-state checkpointer for bit-exact restart (also lets the run be resumed).
    # `cleanup = false` keeps every checkpoint so any one can be picked as an LES IC.
    simulation.output_writers[:chk] = Checkpointer(model; dir, overwrite_existing,
                                                   schedule = TimeInterval(save_interval_3d),
                                                   cleanup = false,
                                                   prefix = file_prefix * "_checkpointer" * rank_suffix)

    return simulation
end

parsing = true

# CLI:
#   julia --project constant_waves.jl Nx Ny Nz Lx Ly Lz ϵ α t₀ W' [stop_time] [Px Py Pz]
#
# Single-GPU example:
#   julia --project constant_waves.jl 768 768 512 0.2 0.2 0.1 0.1 1.2 16.0 0.01 18.0
#
# Multi-GPU (4 ranks, 2x2 horizontal partition) — DeltaAI / GH200:
#   srun -n 4 julia --project constant_waves.jl 768 768 512 0.2 0.2 0.1 0.1 1.2 16.0 0.01 18.0 2 2 1
#
# Pz must currently be 1 (FFT pressure solver requires z-direction to be local).

if parsing
    Nx     = parse(Int,     ARGS[1])
    Ny     = parse(Int,     ARGS[2])
    Nz     = parse(Int,     ARGS[3])
    Lx     = parse(Float64, ARGS[4])
    Ly     = parse(Float64, ARGS[5])
    Lz     = parse(Float64, ARGS[6])
    ϵ      = parse(Float64, ARGS[7])
    α      = parse(Float64, ARGS[8]) * 1e-5
    t₀     = parse(Float64, ARGS[9])
    W′     = parse(Float64, ARGS[10])
    stop_time = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) : 22.0
    Px     = length(ARGS) >= 14 ? parse(Int, ARGS[12]) : 1
    Py     = length(ARGS) >= 14 ? parse(Int, ARGS[13]) : 1
    Pz     = length(ARGS) >= 14 ? parse(Int, ARGS[14]) : 1
end

arch = select_architecture(Px, Py, Pz)
is_root() && @info "Architecture: $arch"

simulation = build_numerical_wave_tank(arch;
                                       Nx, Ny, Nz,
                                       Lx, Ly, Lz,
                                       α, ϵ, t₀, W′, stop_time)

run!(simulation)

if is_root()
    @info "Simulation complete: $simulation. Output:"
    for (name, writer) in simulation.output_writers
        if !(writer isa Checkpointer)
            absfilepath = abspath(writer.filepath)
            @info "OutputWriter $name, $absfilepath:\n $writer"
        end
    end
end

