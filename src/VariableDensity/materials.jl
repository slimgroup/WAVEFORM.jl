export Material, extend_to_pml, extension_matrix

"""
    Material(c, rho)

Velocity `c` (m/s) and density `rho` (kg/m^3) on the physical grid (`nz × nx`).
"""
struct Material
    c::Matrix{Float64}
    rho::Matrix{Float64}
    function Material(c::AbstractMatrix, rho::AbstractMatrix)
        size(c) == size(rho) || throw(DimensionMismatch("c is $(size(c)), rho is $(size(rho))"))
        all(x -> isfinite(x) && x > 0, c) || throw(ArgumentError("velocity must be finite and positive"))
        all(x -> isfinite(x) && x > 0, rho) || throw(ArgumentError("density must be finite and positive"))
        new(Matrix{Float64}(c), Matrix{Float64}(rho))
    end
end

function _extension_index_maps(g::Grid2D)
    nzc, nxc = comp_size(g)
    izmap = clamp.((1:nzc) .- g.npml[1], 1, g.nz)
    ixmap = clamp.((1:nxc) .- g.npml[3], 1, g.nx)
    return izmap, ixmap
end

"""
    extend_to_pml(g, f)

Continue a physical-grid field into the PML by copying the nearest physical node.
"""
function extend_to_pml(g::Grid2D, f::AbstractMatrix)
    size(f) == (g.nz, g.nx) || throw(DimensionMismatch("field is $(size(f)), grid is ($(g.nz), $(g.nx))"))
    izmap, ixmap = _extension_index_maps(g)
    return f[izmap, ixmap]
end

"""
    extension_matrix(g)

Sparse matrix of `extend_to_pml` (`ncomp × nz*nx`). Its adjoint sums PML values onto the
border nodes; it is not a crop.
"""
function extension_matrix(g::Grid2D)
    izmap, ixmap = _extension_index_maps(g)
    cols = vec(LinearIndices((g.nz, g.nx))[izmap, ixmap])
    return sparse(1:ncomp(g), cols, ones(ncomp(g)), ncomp(g), g.nz * g.nx)
end
