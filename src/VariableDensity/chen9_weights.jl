export Chen9Weights, CHEN2013_GLOBAL, CHEN2013_REFINED, FIVE_POINT, points_per_wavelength,
       select_chen_weights, nine_point_symbol, normalized_phase_velocity

"""
    Chen9Weights(β, w1, w2)

Optimized nine-point weights: transverse flux averaging `β` (the paper's `b`) and mass
weights `w0 = 1 - w1 - w2` (paper `c`), `w1` (paper `d`), `w2` (paper `e`).
"""
struct Chen9Weights
    β::Float64
    w0::Float64
    w1::Float64
    w2::Float64
end
Chen9Weights(β, w1, w2) = Chen9Weights(β, 1 - w1 - w2, w1, w2)

"Global optimal weights, Chen et al. (2013) eq. (3.22), for G ∈ [4, 400]."
const CHEN2013_GLOBAL = Chen9Weights(0.7926, 0.3768, -0.0064)
"Standard five-point scheme (ablation only, never a production discretization)."
const FIVE_POINT = Chen9Weights(1.0, 0.0, 0.0)
"Refined optimal weights, Chen et al. (2013) Table 1: `(Gmin, Gmax, weights)`."
const CHEN2013_REFINED = [
    (2.5, 3.0, Chen9Weights(0.6803, 0.4444, 0.0008)),
    (3.0, 4.0, Chen9Weights(0.7427, 0.4088, -0.0036)),
    (4.0, 5.0, Chen9Weights(0.7840, 0.3832, -0.0060)),
    (5.0, 6.0, Chen9Weights(0.8020, 0.3712, -0.0072)),
    (6.0, 8.0, Chen9Weights(0.8133, 0.3637, -0.0075)),
    (8.0, 10.0, Chen9Weights(0.8219, 0.3578, -0.0078)),
    (10.0, 400.0, Chen9Weights(0.8271, 0.3540, -0.0080)),
]

"Gridpoints-per-wavelength interval `(Gmin, Gmax)` for velocities and frequencies (Hz) on spacing `h`."
points_per_wavelength(cmin, cmax, fmin, fmax, h) = (cmin / (h * fmax), cmax / (h * fmin))

"""
    select_chen_weights(Gmin, Gmax)

Rule 3.6 of Chen et al. (2013): the refined Table 1 entry whose interval contains all of
`[Gmin, Gmax]` (the lower row wins at a shared endpoint); else the global weights if
`[Gmin, Gmax] ⊆ [4, 400]`; else an `ArgumentError`. Freeze the result for a whole
survey/prior rather than reselecting per frequency.
"""
function select_chen_weights(Gmin::Real, Gmax::Real)
    Gmin <= Gmax || throw(ArgumentError("Gmin = $Gmin exceeds Gmax = $Gmax"))
    for (lo, hi, w) in CHEN2013_REFINED
        lo <= Gmin && Gmax <= hi && return w
    end
    4 <= Gmin && Gmax <= 400 && return CHEN2013_GLOBAL
    throw(ArgumentError("points per wavelength [$Gmin, $Gmax] is outside the validated range of the Chen et al. (2013) weights ([2.5, 400] refined, [4, 400] global); change h or the frequency band"))
end

"""
    nine_point_symbol(w, b, a, ω, h, ξ, η)

Plane-wave symbol of the homogeneous, PML-free action for buoyancy `b` and compressibility
`a`, with `ξ = kx h`, `η = kz h` (spec Sec. 4).
"""
function nine_point_symbol(w::Chen9Weights, b, a, ω, h, ξ, η)
    stiff = 4b / h^2 * (sin(ξ / 2)^2 * (w.β + (1 - w.β) * cos(η)) + sin(η / 2)^2 * (w.β + (1 - w.β) * cos(ξ)))
    mass = w.w0 + w.w1 / 2 * (cos(ξ) + cos(η)) + w.w2 * cos(ξ) * cos(η)
    return stiff - ω^2 * a * mass
end

"""
    normalized_phase_velocity(w, G, θ)

Numerical-to-exact phase-velocity ratio for `G` gridpoints per wavelength and propagation
angle `θ` (Chen et al. 2013, eq. 3.12), from `nine_point_symbol = 0`.
"""
function normalized_phase_velocity(w::Chen9Weights, G::Real, θ::Real)
    kh = 2π / G
    ξ, η = kh * sin(θ), kh * cos(θ)
    stiff = nine_point_symbol(w, 1.0, 0.0, 0.0, 1.0, ξ, η)       # h = 1, b = 1, mass off
    mass = -nine_point_symbol(w, 0.0, 1.0, 1.0, 1.0, ξ, η)       # b = 0, a = ω = 1
    return sqrt(stiff / mass) / kh
end
