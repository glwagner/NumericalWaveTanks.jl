using GLMakie
using SpecialFunctions
using Printf

f(z, h, β) = 1 + erf(z / h) - 4 * β * Base.exp(z)

z = collect(-10:1e-3:-0.01)

fig = Figure()
ax = Axis(fig[1, 1])
h_slider = Slider(fig[2, 1], range=0.01:0.01:2.0, startvalue=0.01)

h = h_slider.value
β = 0.01

fh = @lift f.(z, $h, 0.05)

z★ = @lift begin
    fh = f.(z, $h, β)
    fmin, k = findmin(fh)
    z[k]
end

z★h = @lift begin
    fh = f.(z, $h, β)
    fmin, k = findmin(fh)
    - z[k] / $h
end

fmin = @lift begin
    fh = f.(z, $h, β)
    minimum(fh)
end

ftxt = @lift begin
    fh = f.(z, $h, β)
    minimum(fh) - 0.1
end
 
lines!(ax, fh, z)
vlines!(ax, 0)
xlims!(ax, -1, 1)

#label = @lift @sprintf("%.2f, %.2f", $h, $z★)
label = @lift @sprintf("%.2f, %.2f", $z★, $z★h)
text!(ax, ftxt, z★, text=label, align=(:right, :center))

display(fig)

