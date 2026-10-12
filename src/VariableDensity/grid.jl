export Grid2D, comp_size, ncomp, physical_z_range, physical_x_range, comp_z, comp_x, z_index, x_index

"""
    Grid2D(nz, nx, h, oz, ox, npml)

Equal-spacing computational grid: `nz × nx` physical nodes with spacing `h` (m), physical
node (1,1) at `(oz, ox)`, plus `npml = (z_lo, z_hi, x_lo, x_hi)` PML nodes on each side.
Arrays on the computational grid are indexed `[iz, ix]` (z fastest, column-major). Outer
closure: homogeneous Dirichlet ghost nodes just outside the computational grid.
"""
struct Grid2D
    nz::Int
    nx::Int
    h::Float64
    oz::Float64
    ox::Float64
    npml::NTuple{4,Int}
    function Grid2D(nz, nx, h, oz, ox, npml)
        nz >= 3 && nx >= 3 || throw(ArgumentError("need at least 3 physical nodes per direction, got ($nz, $nx)"))
        h > 0 || throw(ArgumentError("grid spacing must be positive, got $h"))
        all(>=(0), npml) || throw(ArgumentError("PML widths must be nonnegative, got $npml"))
        new(nz, nx, Float64(h), Float64(oz), Float64(ox), Tuple(Int.(npml)))
    end
end

comp_size(g::Grid2D) = (g.nz + g.npml[1] + g.npml[2], g.nx + g.npml[3] + g.npml[4])
ncomp(g::Grid2D) = prod(comp_size(g))
physical_z_range(g::Grid2D) = (g.npml[1] + 1):(g.npml[1] + g.nz)
physical_x_range(g::Grid2D) = (g.npml[3] + 1):(g.npml[3] + g.nx)
"Physical z coordinate of fractional computational index `ξ`."
comp_z(g::Grid2D, ξ) = g.oz + (ξ - (g.npml[1] + 1)) * g.h
"Physical x coordinate of fractional computational index `ξ`."
comp_x(g::Grid2D, ξ) = g.ox + (ξ - (g.npml[3] + 1)) * g.h
"Fractional computational index of physical coordinate `z`."
z_index(g::Grid2D, z) = (z - g.oz) / g.h + g.npml[1] + 1
"Fractional computational index of physical coordinate `x`."
x_index(g::Grid2D, x) = (x - g.ox) / g.h + g.npml[3] + 1
