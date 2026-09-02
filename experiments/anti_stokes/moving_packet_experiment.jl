#####
##### One member of the paired moving-packet experiment (campaign document, section 9):
#####
#####   quiescent_control  — no turbulence, no packet (regression test: stays at rest)
#####   packet_null        — no turbulence, moving packet, packet-balanced initialization
#####   turbulence_control — ambient turbulence checkpoint, no packet
#####   packet_turbulence  — same checkpoint, moving packet, packet-balanced initialization
#####
##### All members of a pair share the grid, the fixed timestep, the numerics, and the
##### output schedule, so that differencing is exact in space and time.
#####

include("generate_turbulence.jl")

function run_member(; case_name = "1.D",
                      member = "packet_null",
                      level = "S0",
                      seed = 0,
                      Δt = 0.02,
                      numerics = "weno",
                      FT = Float32,
                      arch = GPU(),
                      Lx = 12.0,
                      Ly = 0.8,
                      x_FOV = Lx / 2,
                      σ_upstream = 4,
                      stop_time = nothing,
                      output_interval = 0.1,
                      snapshot_widths = (1, 3, 4, 5, 7, 8),
                      remove_mean_transport = true,
                      animation_slices = false,
                      root = default_data_root(),
                      overwrite = true,
                      progress_interval = 50,
                      tag_extra = "")

    validate_member(member)
    case = anti_stokes_case(case_name, FT)
    (; Nx, Ny, Nz) = level_size(level)
    num = numerics_settings(numerics, FT)

    dir = run_directory(root, case, level, member; seed, Δt, numerics, Lx, Ly, extra=tag_extra)
    mkpath(dir)
    @info "Member $member for case $case_name at level $level → $dir"

    #####
    ##### Grid, packet, model
    #####

    grid = build_grid(arch, FT; Nx, Ny, Nz, Lx, Ly, Lz=case.h, halo=num.halo)
    @info "Grid: $(summary(grid))"

    packet = packet_parameters(case, Lx, x_FOV; σ_upstream)
    stokes_drift = has_packet(member) ? StokesDrift(; ∂z_uˢ, ∂t_uˢ, ∂x_wˢ, ∂t_wˢ, parameters=packet) : nothing

    model = build_model(grid; stokes_drift, advection=num.advection, closure=num.closure)
    @info "Model: $(summary(model))"

    τ₀ = Float64(case.τ₀)
    t_peak = Float64(packet_peak_time(packet))
    isnothing(stop_time) && (stop_time = Float64(packet_stop_time(packet)))

    #####
    ##### Initial condition: uᴸ₀ = uᴱ_turb + uˢ₀, then pressure projection in set!
    #####

    u₀ = XFaceField(grid)
    v₀ = YFaceField(grid)
    w₀ = ZFaceField(grid)

    if has_packet(member)
        set!(u₀, (x, y, z) -> uˢ(x, y, z, 0, packet))
        set!(w₀, (x, y, z) -> wˢ(x, y, z, 0, packet))
    end

    ic_path = ""
    ic_checksum = ""
    ic_metadata = nothing

    if has_turbulence(member)
        ic_path = initial_condition_path(root, case, level, seed; Lx, Ly)
        isfile(ic_path) || error("Initial condition $ic_path not found. Generate it with\n" *
                                 "  julia --project=. experiments/anti_stokes/generate_turbulence.jl " *
                                 "case=$case_name level=$level seed=$seed")
        uₜ, vₜ, wₜ, ic_metadata = load_initial_condition(ic_path)
        size(uₜ) == (Nx, Ny, Nz) || error("Initial condition size $(size(uₜ)) does not match grid $((Nx, Ny, Nz))")
        ic_checksum = file_sha256(ic_path)
        interior(u₀) .+= on_architecture(arch, FT.(uₜ))
        interior(v₀) .+= on_architecture(arch, FT.(vₜ))
        interior(w₀) .+= on_architecture(arch, FT.(wₜ))
        @info "Loaded turbulence checkpoint $ic_path (sha256 $(first(ic_checksum, 12)))"
    end

    # Remove the domain-mean horizontal velocity. In a periodic tank the depth-integrated
    # Lagrangian transport ∫uᴸdz is independent of x and conserved, so without this step the
    # packet's mean Stokes transport would be balanced by a uniform Eulerian current
    # Q/h = ⟨uˢ₀⟩ (≈ 1.4 mm/s for case 1.D) everywhere, instead of by a return flow
    # localized under the packet as in a long laboratory tank. The pressure projection
    # cannot remove a uniform flow, so it is removed here. For turbulent members this also
    # enforces zero mean momentum; the packet/control difference is unaffected up to a
    # constant, which the packet_null member reproduces exactly.
    if remove_mean_transport
        ū = volume_mean(u₀)
        v̄ = volume_mean(v₀)
        interior(u₀) .-= ū
        interior(v₀) .-= v̄
        @info @sprintf("Removed volume-mean velocity (ū, v̄) = (%.3e, %.3e) m s⁻¹", ū, v̄)
    end

    set!(model; u=u₀, v=v₀, w=w₀)
    rms₀ = component_rms(model)
    @info @sprintf("Initial rms (Lagrangian): (%.4e, %.4e, %.4e) m s⁻¹", rms₀...)

    #####
    ##### Simulation
    #####

    simulation = Simulation(model; Δt, stop_time, verbose=false)
    simulation.callbacks[:progress] = Callback(make_progress(has_packet(member) ? packet : nothing),
                                               IterationInterval(progress_interval))

    #####
    ##### Output
    #####

    u, v, w = model.velocities

    n_out = max(1, round(Int, output_interval / Δt))
    abs(n_out * Δt - output_interval) < 1e-8 ||
        @warn "output_interval $output_interval is not a multiple of Δt = $Δt; using $(n_out * Δt) s"

    # y-averaged x-z means and second moments (section 10.1)
    U  = Field(Average(u, dims=2))
    V  = Field(Average(v, dims=2))
    W  = Field(Average(w, dims=2))
    UU = Field(Average(u * u, dims=2))
    VV = Field(Average(v * v, dims=2))
    WW = Field(Average(w * w, dims=2))
    uw = @at (Center, Center, Center) u * w
    UW = Field(Average(uw, dims=2))

    simulation.output_writers[:y_averages] =
        JLD2Writer(model, (; U, V, W, UU, VV, WW, UW); dir,
                   filename = "y_averages",
                   schedule = IterationInterval(n_out),
                   overwrite_existing = overwrite,
                   jld2_kw = JLD2_KW,
                   with_halos = false,
                   array_type = Array{FT})

    # Virtual PIV plane at the grid face nearest x_FOV (section 10.2)
    xf = Array(xnodes(grid, Face()))
    i_FOV = argmin(abs.(xf .- x_FOV))

    simulation.output_writers[:fov_plane] =
        JLD2Writer(model, (; u, v, w); dir,
                   filename = "fov_plane",
                   schedule = IterationInterval(n_out),
                   indices = (i_FOV, :, :),
                   overwrite_existing = overwrite,
                   jld2_kw = JLD2_KW,
                   with_halos = false,
                   array_type = Array{FT})

    # Scalar statistics: packet trajectory check, domain-mean momentum, energy
    # Volume-weighted (Δz-weighted) means and rms on the stretched grid
    Ū, V̄ = Field(Average(u)), Field(Average(v))
    U², V², W² = Field(Average(u * u)), Field(Average(v * v)), Field(Average(w * w))
    scalar(f) = first(Array(interior(compute!(f))))

    statistics = (; uˢ_fov = m -> uˢ(x_FOV, 0, 0, m.clock.time, packet),
                    x_c    = m -> packet_center(m.clock.time, packet),
                    u_mean = m -> scalar(Ū),
                    v_mean = m -> scalar(V̄),
                    u_rms  = m -> sqrt(scalar(U²)),
                    v_rms  = m -> sqrt(scalar(V²)),
                    w_rms  = m -> sqrt(scalar(W²)),
                    w_max  = m -> maximum(abs, m.velocities.w))

    simulation.output_writers[:statistics] =
        JLD2Writer(model, statistics; dir,
                   filename = "statistics",
                   schedule = IterationInterval(n_out),
                   overwrite_existing = overwrite,
                   jld2_kw = JLD2_KW)

    # Sparse three-dimensional snapshots at multiples of τ₀ (section 10.3)
    snapshot_times = [n * τ₀ for n in snapshot_widths if n * τ₀ <= stop_time + 1e-8]
    snapshot_iterations = [round(Int, t / Δt) for t in snapshot_times]

    simulation.output_writers[:snapshots] =
        JLD2Writer(model, (; u, v, w); dir,
                   filename = "snapshots",
                   schedule = SpecifiedIterations(snapshot_iterations),
                   overwrite_existing = overwrite,
                   jld2_kw = JLD2_KW,
                   with_halos = false,
                   array_type = Array{FT})

    # Optional high-cadence slices for animations: an x-z slice through the middle of the
    # tank and the x-y plane one cell below the surface (w) / the top cell (u, v).
    if animation_slices
        simulation.output_writers[:xz_slice] =
            JLD2Writer(model, (; u, v, w); dir,
                       filename = "xz_slice",
                       schedule = IterationInterval(n_out),
                       indices = (:, Ny ÷ 2, :),
                       overwrite_existing = overwrite,
                       jld2_kw = JLD2_KW,
                       with_halos = false,
                       array_type = Array{FT})

        simulation.output_writers[:xy_surface] =
            JLD2Writer(model, (; u, v, w); dir,
                       filename = "xy_surface",
                       schedule = IterationInterval(n_out),
                       indices = (:, :, Nz),
                       overwrite_existing = overwrite,
                       jld2_kw = JLD2_KW,
                       with_halos = false,
                       array_type = Array{FT})
    end

    #####
    ##### Metadata (section 10.4)
    #####

    jldsave(joinpath(dir, "metadata.jld2"), false, IOStream;
            case, member, seed, level, Nx, Ny, Nz, Lx, Ly, Lz = Float64(case.h), FT = string(FT),
            packet, has_packet = has_packet(member), has_turbulence = has_turbulence(member),
            x_FOV, i_FOV, σ_upstream, t_peak, stop_time, τ₀, Δt, output_interval, n_out, remove_mean_transport, animation_slices,
            snapshot_times, snapshot_iterations, numerics,
            advection = summary(model.advection), closure = summary(model.closure),
            oceananigans = oceananigans_version(), commit = git_commit(), dirty = git_dirty(),
            initial_condition = ic_path, initial_condition_sha256 = ic_checksum, ic_metadata,
            rms_initial = rms₀, created = string(now()), hostname = gethostname())

    #####
    ##### Run
    #####

    @info @sprintf("Running to t = %.2f s (packet peak at FOV at t = %.2f s) with Δt = %.4f s...",
                   stop_time, t_peak, Δt)
    wall = time_ns()
    run!(simulation)
    wall_time = 1e-9 * (time_ns() - wall)

    rms₁ = component_rms(model)
    for (name, field) in pairs(model.velocities)
        any(isnan, Array(interior(field))) && error("NaN detected in $name")
    end

    jldsave(joinpath(dir, "run_summary.jld2"), false, IOStream;
            iterations = iteration(simulation), final_time = time(simulation),
            wall_time, seconds_per_iteration = wall_time / max(1, iteration(simulation)),
            rms_initial = rms₀, rms_final = rms₁, completed = string(now()))

    @info @sprintf("Completed %s: %d iterations to t = %.3f s in %s (%.3f s / iteration)",
                   member, iteration(simulation), time(simulation), prettytime(wall_time),
                   wall_time / max(1, iteration(simulation)))

    return simulation, dir
end
