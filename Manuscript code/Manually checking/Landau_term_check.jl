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

"""
Return coefficients for

    E(P) = K * P^2 * (P^2 - amin^2)^2 * (P^2 - bmin^2)^2

This gives equal minima with E = 0 at

    P = 0, +/-amin, +/-bmin

Use a small K first. K = 0.1 is usually much easier to inspect than K = 10.
"""
function coeffs_equal_minima_0ab(amin, bmin; K = 0.1)
    A = amin^2
    B = bmin^2
    return (;
        a = K * A^2 * B^2,
        b = -2K * A * B * (A + B),
        c = K * (A^2 + B^2 + 4A * B),
        d = -2K * (A + B),
        e = K,
    )
end

"""
Return coefficients for

    E(P) = K * (P^2 - amin^2)^2 * (P^2 - bmin^2)^2

This gives four equal minima with E = 0 at

    P = +/-amin, +/-bmin

P = 0 is still a stationary point because this is an even polynomial, but it is
a central barrier rather than one of the minima.
"""
function coeffs_equal_minima_ab(amin, bmin; K = 1.0)
    A = amin^2
    B = bmin^2
    return (;
        a = -2K * A * B * (A + B),
        b = K * (A^2 + B^2 + 4A * B),
        c = -2K * (A + B),
        d = K,
        e = 0.0,
    )
end

# ----------------------------------------------------------------------
# Mode 0: equal-minima design.
#
# This makes E(P) = 0 at P = 0, +/-amin, +/-bmin.
# Recommended for checking "many equal minima" Landau shapes.
# ----------------------------------------------------------------------
# amin = 0.6
# bmin = 1.0
# cs = coeffs_equal_minima_0ab(amin, bmin; K = 500)
# a1, b1, c1, d1, e1 = cs.a, cs.b, cs.c, cs.d, cs.e
# target_x = sort([-bmin, -amin, 0.0, amin, bmin])

# ----------------------------------------------------------------------
# Mode 0b: four equal minima, without P = 0 as a minimum.
#
# This makes E(P) = 0 at +/-amin and +/-bmin.
# P = 0 remains a stationary point, but it is a central barrier.
# Uncomment this block and comment Mode 0 above to use it.
# ----------------------------------------------------------------------
amin = 0.5
bmin = 1.0
cs = coeffs_equal_minima_ab(amin, bmin; K = 100.0)
a1, b1, c1, d1, e1 = cs.a, cs.b, cs.c, cs.d, cs.e
target_x = sort([-bmin, -amin, amin, bmin])

# ----------------------------------------------------------------------
# Mode 1: choose extrema positions directly.
# This is useful if you want many bends but not necessarily equal minima.
#
# The four numbers below mean the curve has stationary points at
# +/-0.32, +/-0.65, +/-1.0, +/-1.32, plus P = 0 automatically.
#
# K controls vertical scale and also flips minima/maxima when its sign changes.
# This choice makes +/-1 local minima, but e1 is negative, so treat this as a
# bounded-window test curve on [-1.5, 1.5], not a globally stable polynomial.
# ----------------------------------------------------------------------
# abs_extrema = [0.55, 1.0, 1.30, 1.40]
# cs = coeffs_from_stationary_abs_positions(abs_extrema; K = 12.0)
# a1, b1, c1, d1, e1 = cs.a, cs.b, cs.c, cs.d, cs.e
# target_x = sort(vcat(-reverse(abs_extrema), 0.0, abs_extrema))

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
# c1 = 10
# d1 = -1
# e1 = 0
# b1 = b_for_extremum_at(a1, c1, d1, e1; target = 1)

a1 = -0.4
b1 = -2.8
c1 = 2
d1 = -1
e1 = 1



Ex1 = range(-1.5, 1.5, length = 1000)
Ex2 = range(-1.0, 1.0, length = 1000)
Ey1 = [landau_energy(x, a1, b1, c1, d1, e1) for x in Ex1]
Ey2 = [landau_energy(x, a1, b1, c1, d1, e1) for x in Ex2]

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
println("target minima/points = ", target_x)
println("E(targets)  = ", [landau_energy(x, a1, b1, c1, d1, e1) for x in target_x])
println("E'(targets) = ", [landau_d1(x, a1, b1, c1, d1, e1) for x in target_x])

fig = Figure(size = (1000, 400))

ax1 = Axis(fig[1, 1], xlabel = "P", ylabel = "Landau energy", title = "Ex1 / Ey1")
lines!(ax1, Ex1, Ey1)
scatter!(ax1, target_x, target_y, color = :red, markersize = 10)

ax2 = Axis(fig[1, 2], xlabel = "P", ylabel = "Landau energy", title = "Ex2 / Ey2")
lines!(ax2, Ex2, Ey2)
scatter!(ax2, target_x, target_y, color = :red, markersize = 10)

display(fig)
f = fig
