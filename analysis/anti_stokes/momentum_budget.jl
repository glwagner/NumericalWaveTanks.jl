#####
##### Local mean-momentum budget of the paired residual at the observation plane
##### (campaign document, 11.4). With ΔU = (U_pt − U_t) − (U_pn − U_n) and Δ applied to
##### every y-averaged moment, the y-averaged x-momentum equation gives
#####
#####     ∂t ΔU  ≈  −∂z Δ⟨u'w'⟩  −  ∂x Δ⟨u'u'⟩  −  (mean advection, pressure, vortex-force terms)
#####
##### The paper's homogeneous approximation keeps only the first term on the right. This
##### script compares ∂t ΔU with −∂z Δ⟨u'w'⟩ and −∂x Δ⟨u'u'⟩ at the FOV and reports the
##### residual, which contains the terms not available from the saved moments.
#####
##### Usage: julia --project=. analysis/anti_stokes/momentum_budget.jl packet=<dir> control=<dir>
#####            [null=<dir> quiescent=<dir> output=<png>]
#####

using CairoMakie
include("common.jl")

args = parse_key_value_args(ARGS)
ΔU, pk, ct = paired_residual(args["packet"], args["control"], get(args, "null", nothing), get(args, "quiescent", nothing))

t, x, z, zf = times(pk), xnodes_faces(pk), znodes_centers(pk), znodes_faces(pk)
Δz = Δz_centers(pk)
τ, tp, i, k = τ₀(pk), t_peak(pk), fov_index(pk), k₀(pk)
Δx = pk.meta["Lx"] / pk.meta["Nx"]
Nz = length(z)
p = run_packet(pk)

m_pk, m_ct = central_moments(pk), central_moments(ct)
Δuw = m_pk.uw .- m_ct.uw   # cell centers
Δuu = m_pk.uu .- m_ct.uu   # x-faces

# Budget terms as full (Nx, Nz, Nt) fields, then wake-age composites (the single-point FOV
# budget is dominated by decorrelation noise; the composite averages ~60 integral scales).
Nx = size(ΔU, 1)
Δt_out = t[2] - t[1]

∂tΔU = similar(ΔU)
∂tΔU[:, :, 2:end-1] = (ΔU[:, :, 3:end] .- ΔU[:, :, 1:end-2]) ./ (2Δt_out)
∂tΔU[:, :, 1] = (ΔU[:, :, 2] .- ΔU[:, :, 1]) ./ Δt_out
∂tΔU[:, :, end] = (ΔU[:, :, end] .- ΔU[:, :, end-1]) ./ Δt_out

# −∂z Δ⟨u'w'⟩: uw at cell centres (Center, Center) → interpolate to faces (zero flux at top and
# bottom) → difference back to centres; average the two cells adjacent to each x-face of ΔU.
Δuw_face_x = 0.5 .* (Δuw .+ circshift(Δuw, (1, 0, 0)))          # (Nx, Nz, Nt) at x-faces
∂zΔuw = zeros(size(ΔU))
for n in axes(ΔU, 3), i in 1:Nx
    flux = zeros(Nz + 1)
    flux[2:Nz] = 0.5 .* (Δuw_face_x[i, 1:Nz-1, n] .+ Δuw_face_x[i, 2:Nz, n])
    ∂zΔuw[i, :, n] = -(flux[2:Nz+1] .- flux[1:Nz]) ./ Δz
end

# −∂x Δ⟨u'u'⟩ at x-faces (uu lives on x-faces; centred over the neighbouring faces)
∂xΔuu = -(circshift(Δuu, (-1, 0, 0)) .- circshift(Δuu, (1, 0, 0))) ./ (2Δx)

residual = ∂tΔU .- ∂zΔuw .- ∂xΔuu

terms = (("∂t ΔU", ∂tΔU), ("−∂z Δ⟨u'w'⟩", ∂zΔuw), ("−∂x Δ⟨u'u'⟩", ∂xΔuu), ("residual (pressure, vortex force, mean advection)", residual))
age_edges = default_age_edges(τ)
composites = [(name, wake_age_composite(A, x, t, p; age_edges)) for (name, A) in terms]
ages = composites[1][2][1]

println("Momentum budget of ΔU, wake-age composites, top five cells (units 1e-6 m/s²):")
for (a₀, a₁) in ((-1, 0), (0, 1), (1, 2))
    @printf("  age %d–%d τ₀:   k₀z      ∂tΔU    −∂zΔu'w'   −∂xΔu'u'   residual\n", a₀, a₁)
    profs = [composite_profile(c[1], c[2], τ, a₀ + 0.125, a₁ - 0.125) for (_, c) in composites]
    for kk in Nz:-1:Nz-4
        @printf("              %6.2f  %8.2f  %9.2f  %9.2f  %9.2f\n", k * z[kk], (1e6 .* getindex.(profs, kk))...)
    end
end

# Single-point FOV comparison for reference (noisy)
passage = findall(τn -> passage_window(pk)[1] - 1e-6 <= τn <= passage_window(pk)[2] + 1e-6, t)
@printf("  FOV single point, passage window rms over depth: ∂tΔU %.2e, −∂zΔu'w' %.2e, −∂xΔu'u' %.2e, residual %.2e m/s²\n",
        sqrt(mean(abs2, ∂tΔU[i, :, passage])), sqrt(mean(abs2, ∂zΔuw[i, :, passage])),
        sqrt(mean(abs2, ∂xΔuu[i, :, passage])), sqrt(mean(abs2, residual[i, :, passage])))

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1800, 1800))
Label(fig[0, 1:2], "Mean-momentum budget of ΔU_turb, wake-age composites: $(args["packet"])", fontsize=20)

lim = 1e6 .* symmetric_range(cat(composites[1][2][2], composites[2][2][2]; dims=3))
for (n, (name, (ages, C, _))) in enumerate(composites)
    ax = Axis(fig[1 + (n - 1) ÷ 2, 1 + (n - 1) % 2]; xlabel="age (x_c − x)/cᵍ (τ₀)", ylabel="k₀ z", title="$name (1e-6 m/s²)")
    hm = heatmap!(ax, ages ./ τ, k .* z, 1e6 .* permutedims(C); colormap=:balance, colorrange=lim)
    vlines!(ax, [0]; color=(:black, 0.5), linestyle=:dot)
    ylims!(ax, -4, 0)
    n == 2 && Colorbar(fig[1:2, 3], hm; width=20)
end

ax_p = Axis(fig[3, 1]; xlabel="1e-6 m/s²", ylabel="k₀ z", title="composite profiles, age −1…0 τ₀ (packet arriving)")
ax_q = Axis(fig[3, 2]; xlabel="1e-6 m/s²", ylabel="k₀ z", title="composite profiles, age 0…1 τ₀ (packet leaving)")
for (name, (ages, C, _)) in composites
    lines!(ax_p, 1e6 .* composite_profile(ages, C, τ, -0.875, -0.125), k .* z; label=name, linewidth=2)
    lines!(ax_q, 1e6 .* composite_profile(ages, C, τ, 0.125, 0.875), k .* z; label=name, linewidth=2)
end
for ax in (ax_p, ax_q)
    vlines!(ax, [0]; color=:black)
    ylims!(ax, -4, 0)
    axislegend(ax; position=:rb, labelsize=12)
end

ax_s = Axis(fig[4, 1:2]; xlabel="age (τ₀)", ylabel="mm/s", title="composite surface ΔU and ∫ΔU dz / h")
Cu = wake_age_composite(ΔU, x, t, p; age_edges)[2]
lines!(ax_s, ages ./ τ, 1e3 .* Cu[end, :]; label="surface ΔU", linewidth=2)
lines!(ax_s, ages ./ τ, 1e3 .* [sum(Cu[:, a] .* Δz) for a in axes(Cu, 2)] ./ sum(Δz); label="∫ΔU dz / h", linewidth=2)
vlines!(ax_s, [0]; color=(:black, 0.5), linestyle=:dot)
axislegend(ax_s; position=:rt)

output = get(args, "output", joinpath(figure_directory(), "momentum_budget_$(case_dirname(run_case(pk)))_$(pk.meta["level"])_seed$(pk.meta["seed"]).png"))
save(output, fig)
@info "Saved $output"
