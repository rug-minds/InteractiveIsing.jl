function weightfunc1(; dc::T) where {T}
    return one(T) / norm2(dc)
end

function weightfunc2(; dc)
    dx, dy, dz = dc
    physical_dr2 = sqrt((0.05 * dx)^2 + (0.05 * dy)^2 + (0.2 * dz)^2)
    return 1 / physical_dr2
end

function weightfunc3(; dc)
    dx, dy, dz = dc
    physical_dr2 = sqrt((0.3 * dx)^2 + (0.3 * dy)^2 + (0.3 * dz)^2)
    prefac = (dx == 0 && dy == 0) || dx == 0 ? 1 : -1
    return prefac / physical_dr2
end

function weightfunc_angle_anti(; dc::DC) where {DC}
    dx, dy, dz = dc
    ax, ay, az = 0.2, 0.2, 0.1
    rx, ry, rz = ax * dx, ay * dy, az * dz
    r = sqrt(rx^2 + ry^2 + rz^2)
    prefac = -1 + 3 * (rz / r)^2
    return prefac / r^3
end

function weightfunc_angle_ferro(; dc)
    dx, dy, dz = dc
    ax, ay, az = 0.2, 0.2, 0.1
    rx, ry, rz = ax * dx, ay * dy, az * dz
    r = sqrt(rx^2 + ry^2 + rz^2)
    prefac = -1 + 3 * (rz / r)^2
    return abs(prefac) / r^3
end

function weightfunc_shell(ax, ay, az, csr, lambda1, lambda2; dc)
    dx, dy, dz = dc
    k1 = 1.0
    k2 = lambda1 * k1
    k3 = lambda2 * k2

    s = dx * dx + dy * dy + dz * dz
    prefac_sr = if s == 1
        k1
    elseif s == 2
        k2
    elseif s == 3
        k3
    else
        0.0
    end

    return csr * prefac_sr
end

function weightfunc_skymion(; dc)
    dx, dy, _ = dc
    prefac = (abs(dy) > 0 || abs(dx) > 0) ? -2 : 2
    return prefac / norm2(dc)
end

function weightfunc_xy_antiferro(ax, ay, az; dc)
    dx, dy, dz = dc
    physical_dr2 = sqrt((ax * dx)^2 + (ay * dy)^2 + (az * dz)^2)
    prefac = (dx == 0 && dy == 0) || dx == 0 ? 1 : -1
    return prefac / physical_dr2
end

function weightfunc_xy_dilog_antiferro(; dc)
    dx, dy, _ = dc
    prefac = (abs(dx) + abs(dy)) % 2 == 0 ? 1.0 : -1.0
    return prefac / norm2(dc)
end

function weightfunc4(; dc)
    return -1 / norm2(dc)
end
