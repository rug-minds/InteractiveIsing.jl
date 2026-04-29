using CairoMakie, FileIO

CairoMakie.activate!()

function newmakie(makietype, args...; kwargs...)
    f = makietype(args...; kwargs...)
    display(f)
    return f
end

"""
Return b for

    E(P) = a*P^2 + b*P^4 + c*P^6 + d*P^8 + e*P^10

such that P = +/- target is an extremum.
"""
function b_for_extremum_at(a, c, d, e; target = 1)
    x2 = target^2
    return -(2a + 6c * x2^2 + 8d * x2^3 + 10e * x2^4) / (4x2)
end

function landau_energy(P, a, b, c, d, e)
    return a * P^2 + b * P^4 + c * P^6 + d * P^8 + e * P^10
end

function landau_d1(P, a, b, c, d, e)
    return 2a * P + 4b * P^3 + 6c * P^5 + 8d * P^7 + 10e * P^9
end

function landau_d2(P, a, b, c, d, e)
    return 2a + 12b * P^2 + 30c * P^4 + 56d * P^6 + 90e * P^8
end

"""
Build a 10th-order even Landau polynomial by choosing the nonzero extrema.

For

    E(P) = a*P^2 + b*P^4 + c*P^6 + d*P^8 + e*P^10

the derivative is

    E'(P) = P * Q(P^2)

so a 10th-order even polynomial can place four positive nonzero
stationary positions, mirrored automatically at +/-P.
"""
function coeffs_from_stationary_abs_positions(abs_positions; K = 1.0)
    length(abs_positions) == 4 || error("Need exactly four positive positions for a 10th-order even polynomial.")

    roots = Float64.(abs_positions) .^ 2
    poly = [1.0]

    for r in roots
        newpoly = zeros(Float64, length(poly) + 1)
        for i in eachindex(poly)
            newpoly[i] += -r * poly[i]
            newpoly[i + 1] += poly[i]
        end
        poly = newpoly
    end

    q0, q1, q2, q3, q4 = K .* poly
    return (; a = q0 / 2, b = q1 / 4, c = q2 / 6, d = q3 / 8, e = q4 / 10)
end

# ----------------------------------------------------------------------
# Mode 1: choose extrema positions directly.
# This is the easiest way to force many bends.
#
# The four numbers below mean the curve has stationary points at
# +/-0.32, +/-0.65, +/-1.0, +/-1.32, plus P = 0 automatically.
#
# K controls vertical scale and also flips minima/maxima when its sign changes.
# This choice makes +/-1 local minima, but e1 is negative, so treat this as a
# bounded-window test curve on [-1.5, 1.5], not a globally stable polynomial.
# ----------------------------------------------------------------------
abs_extrema = [0.55, 1.0, 1.30, 1.40]
cs = coeffs_from_stationary_abs_positions(abs_extrema; K = -12.0)
a1, b1, c1, d1, e1 = cs.a, cs.b, cs.c, cs.d, cs.e

# ----------------------------------------------------------------------
# Mode 1b: if you want e1 > 0 and +/-1 still local minima, use this instead.
# It has fewer wiggles inside |P| < 1, but is stable at large |P|.
# ----------------------------------------------------------------------
# abs_extrema = [0, 1.0, 1.42]
# cs = coeffs_from_stationary_abs_positions(abs_extrema; K = 12.0)
# a1, b1, c1, d1, e1 = cs.a, cs.b, cs.c, cs.d, cs.e

# ----------------------------------------------------------------------
# Mode 2: old manual mode. Tune a,c,d,e; b is solved so +/-1 is stationary.
# ----------------------------------------------------------------------
# a1 = -2
# c1 = -2
# d1 = 1
# e1 = 2
# b1 = b_for_extremum_at(a1, c1, d1, e1; target = 1)

Ex = range(-1.5, 1.5, length = 1000)
Ey = [landau_energy(x, a1, b1, c1, d1, e1) for x in Ex]

target_x = sort(vcat(-reverse(abs_extrema), 0.0, abs_extrema))
target_y = [landau_energy(x, a1, b1, c1, d1, e1) for x in target_x]

println("a1 = ", a1)
println("b1 = ", b1)
println("c1 = ", c1)
println("d1 = ", d1)
println("e1 = ", e1)
println("E'(1)  = ", landau_d1(1, a1, b1, c1, d1, e1))
println("E''(1) = ", landau_d2(1, a1, b1, c1, d1, e1), "  # > 0 means +/-1 are local minima")
println("E(0)   = ", landau_energy(0, a1, b1, c1, d1, e1))
println("E(1)   = ", landau_energy(1, a1, b1, c1, d1, e1))
println("stationary |P| targets = ", abs_extrema)
println("E'(targets) = ", [landau_d1(x, a1, b1, c1, d1, e1) for x in target_x])

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "P", ylabel = "Landau energy")
lines!(ax, Ex, Ey)
scatter!(ax, target_x, target_y, color = :red, markersize = 10)
display(fig)
f1 = fig
