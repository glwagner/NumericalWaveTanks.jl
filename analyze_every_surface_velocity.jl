using MAT
using GLMakie
using Printf

filename = "data/every_surface_velocity.mat"

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Time (s)", ylabel="Surface velocity (m s⁻¹)")

vars = matread(filename)

ρw = 1000.0
ν = 1.05e-6

tm = [
    [70, 80],
    [83, 98],
    [96, 113],
    [121, 137],
]

Um = [
    [0.06, 0.18],
    [0.03, 0.18],
    [0.06, 0.18],
    [0.09, 0.18],
]

colors = []
βs = []

for (t, U) in zip(tm, Um)
    ln = lines!(ax, t, U, linewidth=5)
    dUdt = (U[2] - U[1]) / (t[2] - t[1])
    β = dUdt * 2 * sqrt(ν / π)
    push!(colors, ln.color)
    push!(βs, β)
end

for (n, exp) in enumerate(["1", "2", "3", "4"])
    U = vars["BIN"]["R" * exp]["U"][:]
    t = vars["BIN"]["R" * exp]["time"][:]
    β = βs[n]
    label = @sprintf("Experiment %s, β = %.1e", exp, β)
    scatter!(ax, t, U, color=(colors[n], 0.6); label)
end


axislegend(ax)
display(fig)

