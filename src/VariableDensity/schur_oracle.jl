export SchurReduction, schur_reduction, reduced_data, reconstruct_field

"""
    SchurReduction

Dense exact elimination of the `f` unknowns (spec Sec. 5.2) for tiny problems:
`S = A_cc - A_cf A_ff⁻¹ A_fc`, `B_eff = B_c - A_cf A_ff⁻¹ B_f`, `R_eff = R_c - R_f A_ff⁻¹ A_fc`,
`D_direct = R_f A_ff⁻¹ B_f`. Every inverse is a sparse-LU solve.
"""
struct SchurReduction{T}
    fine::FineBlockLU{T}
    S::Matrix{ComplexF64}
    Beff::Matrix{ComplexF64}
    Reff::Matrix{ComplexF64}
    Ddirect::Matrix{ComplexF64}
    Bf::Matrix{ComplexF64}
    Afc::SparseMatrixCSC{ComplexF64,Int}
end

"""
    schur_reduction(A, B, R, s; rcond_min=1e-8)

Exact Schur reduction of the sparse system `A`, sources `B` (`N × n_s`) and receivers `R`
(`n_r × N`). Throws `IllConditionedFineBlock` (see `FineBlockLU`).
"""
function schur_reduction(A::SparseMatrixCSC, B::AbstractMatrix, R::AbstractMatrix, s::CoarseFineSplit; rcond_min::Real=1e-8)
    N = length(s)
    size(B, 1) == N || throw(DimensionMismatch("B has $(size(B, 1)) rows, the split has $N unknowns"))
    size(R, 2) == N || throw(DimensionMismatch("R has $(size(R, 2)) columns, the split has $N unknowns"))
    fb = FineBlockLU(A, s; rcond_min)
    ic, jf = s.ic, s.jf
    Bm, Rm = Matrix{ComplexF64}(B), Matrix{ComplexF64}(R)
    Acc, Acf, Afc = block_matrix(A, s, :c, :c), block_matrix(A, s, :c, :f), block_matrix(A, s, :f, :c)
    if isempty(jf)
        return SchurReduction(fb, Matrix{ComplexF64}(Acc), Bm[ic, :], Rm[:, ic],
                              zeros(ComplexF64, size(Rm, 1), size(Bm, 2)), Bm[jf, :], SparseMatrixCSC{ComplexF64,Int}(Afc))
    end
    X = fb.F \ Matrix{ComplexF64}(Afc)                  # A_ff⁻¹ A_fc
    Y = fb.F \ Bm[jf, :]                                 # A_ff⁻¹ B_f
    return SchurReduction(fb, Matrix{ComplexF64}(Acc) - Acf * X, Bm[ic, :] - Acf * Y,
                          Rm[:, ic] - Rm[:, jf] * X, Rm[:, jf] * Y, Bm[jf, :], SparseMatrixCSC{ComplexF64,Int}(Afc))
end

"Data `F q = R_eff S⁻¹ B_eff q + D_direct q`."
reduced_data(sr::SchurReduction, q::AbstractVecOrMat) = sr.Reff * (lu(sr.S) \ (sr.Beff * q)) + sr.Ddirect * q

"Full field from the dense reduction: `u_c = S⁻¹ B_eff q`, `u_f = A_ff⁻¹ (B_f q - A_fc u_c)`."
function reconstruct_field(sr::SchurReduction, q::AbstractVecOrMat)
    s = sr.fine.split
    uc = lu(sr.S) \ (sr.Beff * q)
    u = zeros(ComplexF64, length(s), size(uc, 2))
    u[s.ic, :] .= uc
    isempty(s.jf) || (u[s.jf, :] .= sr.fine.F \ (sr.Bf * q - sr.Afc * uc))
    return q isa AbstractVector ? vec(u) : u
end
