export CoarseFineSplit, build_coarse_fine_maps, restriction, injection, split_summary

"""
    CoarseFineSplit(grid_size, retained)

Disjoint retained (`c`) and eliminated (`f`) index sets over all `N = prod(grid_size)`
computational unknowns (spec Sec. 5.1), from a mask over the column-major `[iz, ix]` ordering.
"""
struct CoarseFineSplit
    grid_size::Tuple{Int,Int}
    retained::BitVector
    ic::Vector{Int}
    jf::Vector{Int}
    function CoarseFineSplit(grid_size::Tuple{Int,Int}, retained::AbstractVector{Bool})
        length(retained) == prod(grid_size) ||
            throw(DimensionMismatch("mask has $(length(retained)) entries, grid has $(prod(grid_size))"))
        r = BitVector(retained)
        new(grid_size, r, findall(r), findall(!, r))
    end
end

Base.length(s::CoarseFineSplit) = length(s.retained)

_block_indices(s::CoarseFineSplit, block::Symbol) =
    block === :c ? s.ic : block === :f ? s.jf : throw(ArgumentError("block must be :c or :f, got :$block"))

"""
    restriction(s, block)

Coordinate-selection restriction `P_c` or `P_f` (`block = :c` or `:f`), a sparse
`N_block × N` matrix whose rows are coordinate vectors.
"""
function restriction(s::CoarseFineSplit, block::Symbol)
    idx = _block_indices(s, block)
    return sparse(1:length(idx), idx, ones(length(idx)), length(idx), length(s))
end

"Injection `E_c = P_c*` or `E_f = P_f*` (`N × N_block`)."
injection(s::CoarseFineSplit, block::Symbol) = sparse(transpose(restriction(s, block)))

"""
    build_coarse_fine_maps(g, support=Int[]; stride=2, retain_pml=true, retain_support=true)

Retained set of spec Sec. 5.3: physical nodes whose physical indices `(iz_p - 1, ix_p - 1)` are
both multiples of `stride` (parity anchored at physical node (1,1), so the choice is a property
of global physical coordinates and does not depend on PML widths or any rank-local layout),
plus all PML nodes when `retain_pml`, plus the computational-grid indices in `support` (e.g.
`acquisition_support(B, R)`) when `retain_support`. Everything else is eliminated.
"""
function build_coarse_fine_maps(g::Grid2D, support::AbstractVector{<:Integer}=Int[];
                                stride::Integer=2, retain_pml::Bool=true, retain_support::Bool=true)
    stride >= 1 || throw(ArgumentError("stride must be ≥ 1, got $stride"))
    nzc, nxc = comp_size(g)
    all(i -> 1 <= i <= nzc * nxc, support) ||
        throw(ArgumentError("support indices must lie in 1:$(nzc * nxc)"))
    zr, xr = physical_z_range(g), physical_x_range(g)
    keep = falses(nzc, nxc)
    for ix in 1:nxc, iz in 1:nzc
        keep[iz, ix] = (iz in zr && ix in xr) ?
            (mod(iz - first(zr), stride) == 0 && mod(ix - first(xr), stride) == 0) : retain_pml
    end
    retain_support && (keep[support] .= true)
    return CoarseFineSplit((nzc, nxc), vec(keep))
end

"""
    split_summary(s, g)

Dimensions of a split on grid `g`: `N`, `N_c`, `N_f`, the retained fraction `N_c/N`, the
number of retained PML nodes and their fraction of `N_c`, and the retained fraction of the
physical nodes.
"""
function split_summary(s::CoarseFineSplit, g::Grid2D)
    s.grid_size == comp_size(g) || throw(DimensionMismatch("split is $(s.grid_size), grid is $(comp_size(g))"))
    keep = reshape(s.retained, s.grid_size)
    phys = view(keep, physical_z_range(g), physical_x_range(g))
    Nc, N = length(s.ic), length(s)
    npml_c = Nc - count(phys)
    return (N=N, N_c=Nc, N_f=length(s.jf), retained_fraction=Nc / N,
            pml_retained=npml_c, pml_fraction_of_retained=npml_c / max(Nc, 1),
            physical_retained_fraction=count(phys) / length(phys))
end
