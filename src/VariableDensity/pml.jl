export Stretch2D, no_stretch, chen_stretch

"""
    Stretch2D(sz, sx)

Complex coordinate stretches `s = 1 + iσ/ω` (exp(-iωt) convention) as callables of a
fractional computational index (`ξ = 3.5` is the face between nodes 3 and 4).
"""
struct Stretch2D{FZ,FX}
    sz::FZ
    sx::FX
end

"No PML: `s ≡ 1`."
no_stretch() = Stretch2D(ξ -> complex(1.0), ξ -> complex(1.0))

"""
    chen_stretch(g, ω; f0, a0=1.79)

PML of Chen et al. (2013), Sec. 2: σ(l) = 2π a0 f0 (l/L)^2, with `l` the distance into the
PML from the outermost physical node and `L` that side's PML thickness. The paper writes
e = 1 - iσ/ω for exp(+iωt); this is the conjugate s = 1 + iσ/ω for exp(-iωt). The legacy
`pml_func1d` divides σ by the frequency in Hz (2π stronger damping) and offsets the
lower-side distance by one cell (NonLocal docs/conventions.md, findings 1-2).
"""
function chen_stretch(g::Grid2D, ω::Real; f0::Real, a0::Real=1.79)
    ω > 0 || throw(ArgumentError("ω must be positive, got $ω"))
    f0 > 0 || throw(ArgumentError("f0 must be positive, got $f0"))
    σmax = 2π * a0 * f0
    function side(nlo, nhi, first, last)
        return function (ξ)
            if ξ < first && nlo > 0
                return 1.0 + im * σmax * ((first - ξ) / nlo)^2 / ω
            elseif ξ > last && nhi > 0
                return 1.0 + im * σmax * ((ξ - last) / nhi)^2 / ω
            else
                return complex(1.0)
            end
        end
    end
    zr, xr = physical_z_range(g), physical_x_range(g)
    return Stretch2D(side(g.npml[1], g.npml[2], first(zr), last(zr)),
                     side(g.npml[3], g.npml[4], first(xr), last(xr)))
end
