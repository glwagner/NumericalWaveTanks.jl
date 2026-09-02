#####
##### Regular-wave cases (Experiments 2 and 3 of Ellingsen et al. 2026; campaign document, 13.4).
#####
##### In the current-following frame a steady wave train is a horizontally uniform, steady Stokes
##### drift uˢ(z) = Uˢ₀ e^{2kz} acting on turbulence that decays as it would while advecting from
##### the grid to the field of view; the FOV corresponds to t_FOV = L_FOV / U₀ after the grid.
##### The box is doubly periodic and horizontally homogeneous, so the primary output is the set
##### of horizontally averaged profiles. Members:
#####
#####   quiescent_control  — no turbulence, no waves (stays at rest)
#####   waves_null         — no turbulence, waves: the closed-channel return flow u_rf and any
#####                        numerical response to the Lagrangian initialization
#####   turbulence_control — turbulence checkpoint, no waves
#####   waves_turbulence   — same checkpoint, waves from t = 0 (Lagrangian initialization
#####                        uᴸ₀ = uᴱ_turb + uˢ with the volume mean removed, which yields u_rf)
#####
##### The turbulence-induced Eulerian change is
#####   ΔU(z, t) = (U_waves+turb − U_turb) − (U_waves+null − U_null).
#####

include("generate_turbulence.jl")

@inline regular_∂z_uˢ(z, t, p) = 2 * p.k * p.Uˢ₀ * exp(2 * p.k * z)
@inline regular_uˢ(z, p) = p.Uˢ₀ * exp(2 * p.k * z)

function run_regular_member(; case_name = "2.A.1.3",
                              member = "waves_null",
                              level = "R1",
                              seed = 0,
                              Δt = 0.02,
                              numerics = "weno",
                              FT = Float32,
                              arch = GPU(),
                              Lx = 3.2,
                              Ly = 3.2,
                              stop_time = nothing,
                              output_interval = 0.1,
                              remove_mean_transport = true,
                              root = default_data_root(),
                              overwrite = true,
                              progress_interval = 50,
                              tag_extra = "")

    validate_member(member)
    has_packet(member) && error("Member $member is a moving-packet member; use run_moving_packet.jl")
    case = anti_stokes_case(case_name, FT)
    is_regular(case) || error("Case $case_name is a wave-group case; use run_moving_packet.jl")
    (; Nx, Ny, Nz) = level_size(level)
    num = numerics_settings(numerics, FT)

    dir = run_directory(root, case, level, member; seed, Δt, numerics, Lx, Ly, extra=tag_extra)
    mkpath(dir)
    @info "Regular-wave member $member for case $case_name at level $level → $dir"

    grid = build_grid(arch, FT; Nx, Ny, Nz, Lx, Ly, Lz=case.h, halo=num.halo)
    @info "Grid: $(summary(grid))"

    waves = (; k = case.k, Uˢ₀ = case.Uˢ₀)
    stokes_drift = has_waves(member) ? UniformStokesDrift(; ∂z_uˢ = regular_∂z_uˢ, parameters = waves) : nothing
    model = build_model(grid; stokes_drift, advection=num.advection, closure=num.closure)

    t_FOV = Float64(case.t_FOV)
    isnothing(stop_time) && (stop_time = 1.25 * t_FOV)

    #####
    ##### Initial condition
    #####

    u₀ = XFaceField(grid)
    v₀ = YFaceField(grid)
    w₀ = ZFaceField(grid)
    has_waves(member) && set!(u₀, (x, y, z) -> regular_uˢ(z, waves))

    ic_path, ic_checksum, ic_metadata = "", "", nothing
    if has_turbulence(member)
        ic_path = initial_condition_path(root, case, level, seed; Lx, Ly)
        isfile(ic_path) || error("Initial condition $ic_path not found. Generate it with\n" *
                                 "  julia --project=. experiments/anti_stokes/generate_turbulence.jl " *
                                 "case=$case_name level=$level seed=$seed Lx=$Lx Ly=$Ly")
        uₜ, vₜ, wₜ, ic_metadata = load_initial_condition(ic_path)
        size(uₜ) == size(interior(u₀)) || error("Initial condition size $(size(uₜ)) does not match the grid")
        ic_checksum = file_sha256(ic_path)
        interior(u₀) .+= on_architecture(arch, FT.(uₜ))
        interior(v₀) .+= on_architecture(arch, FT.(vₜ))
        interior(w₀) .+= on_architecture(arch, FT.(wₜ))
        @info "Loaded turbulence checkpoint $ic_path (sha256 $(first(ic_checksum, 12)))"
    end

    # The volume mean of uˢ is the Stokes transport per unit depth; removing it from the
    # Lagrangian initial condition produces the closed-channel Eulerian return flow u_rf.
    if remove_mean_transport
        ū, v̄ = volume_mean(u₀), volume_mean(v₀)
        interior(u₀) .-= ū
        interior(v₀) .-= v̄
        @info @sprintf("Removed volume-mean velocity (ū, v̄) = (%.3e, %.3e) m s⁻¹ (analytic u_rf = %.3e)", ū, v̄, case.u_rf)
    end

    set!(model; u=u₀, v=v₀, w=w₀)
    rms₀ = component_rms(model)

    #####
    ##### Simulation and output
    #####

    simulation = Simulation(model; Δt, stop_time, verbose=false)
    simulation.callbacks[:progress] = Callback(make_progress(), IterationInterval(progress_interval))

    u, v, w = model.velocities
    n_out = max(1, round(Int, output_interval / Δt))

    U  = Field(Average(u, dims=(1, 2)))
    V  = Field(Average(v, dims=(1, 2)))
    W  = Field(Average(w, dims=(1, 2)))
    UU = Field(Average(u * u, dims=(1, 2)))
    VV = Field(Average(v * v, dims=(1, 2)))
    WW = Field(Average(w * w, dims=(1, 2)))
    uw = @at (Center, Center, Center) u * w
    UW = Field(Average(uw, dims=(1, 2)))

    simulation.output_writers[:profiles] =
        JLD2Writer(model, (; U, V, W, UU, VV, WW, UW); dir,
                   filename = "profiles",
                   schedule = IterationInterval(n_out),
                   overwrite_existing = overwrite,
                   jld2_kw = JLD2_KW,
                   with_halos = false,
                   array_type = Array{FT})

    Ū, V̄ = Field(Average(u)), Field(Average(v))
    U², V², W² = Field(Average(u * u)), Field(Average(v * v)), Field(Average(w * w))
    scalar(f) = first(Array(interior(compute!(f))))
    statistics = (; u_mean = m -> scalar(Ū), v_mean = m -> scalar(V̄),
                    u_rms = m -> sqrt(scalar(U²)), v_rms = m -> sqrt(scalar(V²)), w_rms = m -> sqrt(scalar(W²)),
                    w_max = m -> maximum(abs, m.velocities.w))

    simulation.output_writers[:statistics] =
        JLD2Writer(model, statistics; dir, filename = "statistics", schedule = IterationInterval(n_out),
                   overwrite_existing = overwrite, jld2_kw = JLD2_KW)

    snapshot_times_ = [t for t in (t_FOV / 2, t_FOV, stop_time) if 0 < t <= stop_time + 1e-8]
    snapshot_iterations = [round(Int, t / Δt) for t in snapshot_times_]
    simulation.output_writers[:snapshots] =
        JLD2Writer(model, (; u, v, w); dir, filename = "snapshots",
                   schedule = SpecifiedIterations(snapshot_iterations),
                   overwrite_existing = overwrite, jld2_kw = JLD2_KW, with_halos = false, array_type = Array{FT})

    jldsave(joinpath(dir, "metadata.jld2"), false, IOStream;
            case, member, seed, level, Nx, Ny, Nz, Lx, Ly, Lz = Float64(case.h), FT = string(FT),
            waves, has_waves = has_waves(member), has_turbulence = has_turbulence(member), regular = true,
            t_FOV, stop_time, Δt, output_interval, n_out, remove_mean_transport,
            snapshot_times = snapshot_times_, snapshot_iterations, numerics,
            advection = summary(model.advection), closure = summary(model.closure),
            oceananigans = oceananigans_version(), commit = git_commit(), dirty = git_dirty(),
            initial_condition = ic_path, initial_condition_sha256 = ic_checksum, ic_metadata,
            rms_initial = rms₀, created = string(now()), hostname = gethostname())

    @info @sprintf("Running to t = %.2f s (FOV at t = %.2f s) with Δt = %.4f s...", stop_time, t_FOV, Δt)
    wall = time_ns()
    run!(simulation)
    wall_time = 1e-9 * (time_ns() - wall)

    for (name, field) in pairs(model.velocities)
        any(isnan, Array(interior(field))) && error("NaN detected in $name")
    end
    jldsave(joinpath(dir, "run_summary.jld2"); iterations = iteration(simulation), final_time = time(simulation),
            wall_time, rms_initial = rms₀, rms_final = component_rms(model), completed = string(now()))
    @info @sprintf("Completed %s: %d iterations to t = %.3f s in %s", member, iteration(simulation), time(simulation), prettytime(wall_time))
    return simulation, dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_key_value_args(ARGS)
    stop_time = haskey(args, "stop_time") ? getarg(args, "stop_time", 0.0) : nothing
    run_regular_member(; case_name = getarg(args, "case", "2.A.1.3"),
                         member = getarg(args, "member", "waves_null"),
                         level = getarg(args, "level", "R1"),
                         seed = getarg(args, "seed", 0),
                         Δt = getarg(args, "dt", 0.02),
                         numerics = getarg(args, "numerics", "weno"),
                         FT = float_type(getarg(args, "FT", "Float32")),
                         arch = architecture(getarg(args, "arch", "gpu")),
                         Lx = getarg(args, "Lx", 3.2),
                         Ly = getarg(args, "Ly", 3.2),
                         stop_time,
                         output_interval = getarg(args, "output_interval", 0.1),
                         root = getarg(args, "root", default_data_root()),
                         overwrite = getarg(args, "overwrite", true),
                         progress_interval = getarg(args, "progress_interval", 50),
                         tag_extra = getarg(args, "tag", ""))
end
