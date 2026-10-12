export NINE_POINT_OFFSETS, NinePointStencil, probe_stencil, build_coefficients, apply_A!, apply_Aadj!, assemble_sparse

"Stencil offsets `(dz, dx)`; coefficient `k` multiplies `u[iz+dz, ix+dx]`."
const NINE_POINT_OFFSETS = Tuple{Int,Int}[(dz, dx) for dx in -1:1 for dz in -1:1]

"""
    NinePointStencil(coeffs)

Nine complex coefficient fields `coeffs[iz, ix, k]`, the matrix-free representation of spec
Sec. 6: `(A u)[iz, ix] = Σ_k coeffs[iz, ix, k] u[iz+dz_k, ix+dx_k]`, with zero coefficients
for neighbours outside the grid (Dirichlet closure).
"""
struct NinePointStencil
    coeffs::Array{ComplexF64,3}
end

Base.size(S::NinePointStencil) = (size(S.coeffs, 1), size(S.coeffs, 2))

"""
    probe_stencil(apply, nz, nx)

Recover the nine coefficient fields of a radius-one linear operator from nine colored probes:
in every 3×3 neighbourhood each color `(iz mod 3, ix mod 3)` appears exactly once, so each
output entry of a probe isolates one coefficient.
"""
function probe_stencil(apply, nz::Int, nx::Int)
    C = zeros(ComplexF64, nz, nx, 9)
    for cx in 0:2, cz in 0:2
        e = zeros(ComplexF64, nz, nx)
        e[(1+cz):3:end, (1+cx):3:end] .= 1
        y = apply(e)
        for (k, (dz, dx)) in enumerate(NINE_POINT_OFFSETS),
            ix in max(1, 1 - dx):min(nx, nx - dx), iz in max(1, 1 - dz):min(nz, nz - dz)
            if mod(iz + dz - 1, 3) == cz && mod(ix + dx - 1, 3) == cx
                C[iz, ix, k] = y[iz, ix]
            end
        end
    end
    return NinePointStencil(C)
end

"Nine-point stencil of `A_h`, probed from `reference_apply`."
NinePointStencil(op::HelmholtzPML) = probe_stencil(u -> reference_apply(op, u), size(op.μ)...)

"""
    build_coefficients(material, ω, grid, stretch, weights)

Spec Sec. 12 interface: the nine coefficient fields of the variable-density PML operator.
"""
build_coefficients(m::Material, ω::Real, g::Grid2D, st::Stretch2D, w::Chen9Weights) =
    NinePointStencil(HelmholtzPML(g, m, ω, w, st))

"""
    apply_A!(y, S, u)

`y = A u` for vectors (column-major `[iz, ix]` ordering) or matrices on the grid.
"""
function apply_A!(y::AbstractArray, S::NinePointStencil, u::AbstractArray)
    nz, nx = size(S)
    Y, U = reshape(y, nz, nx), reshape(u, nz, nx)
    fill!(Y, 0)
    for (k, (dz, dx)) in enumerate(NINE_POINT_OFFSETS)
        for ix in max(1, 1 - dx):min(nx, nx - dx), iz in max(1, 1 - dz):min(nz, nz - dz)
            Y[iz, ix] += S.coeffs[iz, ix, k] * U[iz+dz, ix+dx]
        end
    end
    return y
end

"""
    apply_Aadj!(y, S, v)

`y = A* v`, i.e. `(A* v)(x) = Σ_δ conj(h_δ(x - δ)) v(x - δ)` (spec Sec. 6): the coefficient
is taken at the shifted node, not at the output node.
"""
function apply_Aadj!(y::AbstractArray, S::NinePointStencil, v::AbstractArray)
    nz, nx = size(S)
    Y, V = reshape(y, nz, nx), reshape(v, nz, nx)
    fill!(Y, 0)
    for (k, (dz, dx)) in enumerate(NINE_POINT_OFFSETS)
        for ix in max(1, 1 + dx):min(nx, nx + dx), iz in max(1, 1 + dz):min(nz, nz + dz)
            Y[iz, ix] += conj(S.coeffs[iz-dz, ix-dx, k]) * V[iz-dz, ix-dx]
        end
    end
    return y
end

"""
    assemble_sparse(S)

Sparse matrix of the stencil (`N × N`, `N = nz*nx`, column-major node ordering).
"""
function assemble_sparse(S::NinePointStencil)
    nz, nx = size(S)
    L = LinearIndices((nz, nx))
    I, J, V = Int[], Int[], ComplexF64[]
    for (k, (dz, dx)) in enumerate(NINE_POINT_OFFSETS)
        for ix in max(1, 1 - dx):min(nx, nx - dx), iz in max(1, 1 - dz):min(nz, nz - dz)
            c = S.coeffs[iz, ix, k]
            iszero(c) && continue
            push!(I, L[iz, ix]); push!(J, L[iz+dz, ix+dx]); push!(V, c)
        end
    end
    return sparse(I, J, V, nz * nx, nz * nx)
end

"Sparse system matrix of `A_h` (oracle path: definition → probed stencil → sparse)."
assemble_sparse(op::HelmholtzPML) = assemble_sparse(NinePointStencil(op))
