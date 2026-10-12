export Acquisition, interpolation_matrix, receiver_matrix, source_matrix, acquisition_support

"""
    Acquisition(src_z, src_x, rec_z, rec_x)

Source and receiver positions (m); all interpolation support must lie in the physical domain.
"""
struct Acquisition
    src_z::Vector{Float64}
    src_x::Vector{Float64}
    rec_z::Vector{Float64}
    rec_x::Vector{Float64}
    function Acquisition(src_z, src_x, rec_z, rec_x)
        length(src_z) == length(src_x) || throw(DimensionMismatch("source z/x lengths differ"))
        length(rec_z) == length(rec_x) || throw(DimensionMismatch("receiver z/x lengths differ"))
        new(collect(Float64, src_z), collect(Float64, src_x), collect(Float64, rec_z), collect(Float64, rec_x))
    end
end

# Positions within `snap` (in grid cells) of a node, plus a round-off allowance of 16 ulp of the
# coordinate, are treated as on the node, so round-off cannot add a zero-weight neighbour in the
# PML even at field-scale (e.g. UTM) coordinates.
function _bilinear_support(ξ::Real, snap::Real)
    i0 = floor(Int, ξ)
    t = ξ - i0
    t < snap && return [(i0, 1.0)]
    t > 1 - snap && return [(i0 + 1, 1.0)]
    return [(i0, 1 - t), (i0 + 1, t)]
end

"""
    interpolation_matrix(g, zs, xs; snap=1e-9)

Bilinear interpolation from the computational grid to the points (`npts × ncomp`). Throws an
`ArgumentError` if any support node lies outside the physical domain (PML included).
"""
function interpolation_matrix(g::Grid2D, zs::AbstractVector, xs::AbstractVector; snap::Real=1e-9)
    length(zs) == length(xs) || throw(DimensionMismatch("z and x coordinate lengths differ"))
    zr, xr = physical_z_range(g), physical_x_range(g)
    L = LinearIndices(comp_size(g))
    I, J, V = Int[], Int[], Float64[]
    for (p, (z, x)) in enumerate(zip(zs, xs))
        sz = snap + 16 * eps(max(abs(z), abs(g.oz))) / g.h
        sx = snap + 16 * eps(max(abs(x), abs(g.ox))) / g.h
        for (iz, wz) in _bilinear_support(z_index(g, z), sz), (ix, wx) in _bilinear_support(x_index(g, x), sx)
            (iz in zr && ix in xr) || throw(ArgumentError("point $p at (z, x) = ($z, $x) has interpolation support outside the physical domain (PML or beyond)"))
            push!(I, p); push!(J, L[iz, ix]); push!(V, wz * wx)
        end
    end
    return sparse(I, J, V, length(zs), ncomp(g))
end

"Receiver sampling `R` (`nrec × ncomp`)."
receiver_matrix(g::Grid2D, acq::Acquisition; snap::Real=1e-9) = interpolation_matrix(g, acq.rec_z, acq.rec_x; snap)

"""
    source_matrix(g, acq; snap=1e-9)

Source injection `B` (`ncomp × nsrc`): transposed bilinear interpolation scaled by `1/h^2`, a
discrete delta for `-∇·(b∇u) - ω²a u = δ(x - x_s)`. Sources are physical, so `J = 1`.
"""
source_matrix(g::Grid2D, acq::Acquisition; snap::Real=1e-9) =
    sparse(transpose(interpolation_matrix(g, acq.src_z, acq.src_x; snap))) ./ g.h^2

"""
    acquisition_support(B, R)

Sorted computational-grid indices touched by source injection or receiver sampling.
"""
acquisition_support(B::AbstractMatrix, R::AbstractMatrix) =
    sort(union(findnz(sparse(B))[1], findnz(sparse(R))[2]))
