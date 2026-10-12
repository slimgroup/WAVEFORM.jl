export IllConditionedFineBlock, rcond_estimate, FineBlockLU, fine_solve!, fine_solve_adj!, block_matrix, apply_block!

"Raised when the eliminated block `A_ff` is singular or too ill-conditioned to eliminate (spec Sec. 5.3)."
struct IllConditionedFineBlock <: Exception
    rcond::Float64
    threshold::Float64
end
Base.showerror(io::IO, e::IllConditionedFineBlock) =
    print(io, "IllConditionedFineBlock: estimated reciprocal 1-norm condition number of A_ff is ",
          e.rcond, " < ", e.threshold, "; choose another retained set or basis instead of regularizing")

"""
    rcond_estimate(A, F)

Estimate of the reciprocal 1-norm condition number `1 / (‖A‖₁ ‖A⁻¹‖₁)` from a factorization
`F` of `A`, with the Hager–Higham estimator of `‖A⁻¹‖₁` (a lower bound, usually exact), using
only solves with `F` and `F'`. Being a lower bound on `‖A⁻¹‖₁`, it can only overestimate
the reciprocal condition number.
"""
function rcond_estimate(A::AbstractMatrix, F)
    n = size(A, 1)
    n == 0 && return 1.0
    x = fill(complex(1.0 / n), n)
    est = 0.0
    for k in 1:5
        y = F \ x
        est = max(est, norm(y, 1))
        ξ = [iszero(v) ? one(v) : v / abs(v) for v in y]
        z = F' \ ξ
        j = argmax(abs.(z))
        (k > 1 && abs(z[j]) <= real(dot(z, x))) && break
        x = zeros(ComplexF64, n); x[j] = 1
    end
    # Higham's alternative vector guards against the rare underestimate
    alt = [complex((-1.0)^(i + 1) * (1 + (i - 1) / max(n - 1, 1))) for i in 1:n]
    est = max(est, 2 * norm(F \ alt, 1) / (3n))
    return 1 / (opnorm(A, 1) * est)
end

"""
    FineBlockLU(A, s; rcond_min=1e-12)

Exact sparse-LU solver for the eliminated block `A_ff = P_f A E_f` (the tiny-grid oracle of
spec Sec. 8). Throws `IllConditionedFineBlock` if `A_ff` is singular or its estimated
reciprocal condition number is below `rcond_min`; it never regularizes.
"""
struct FineBlockLU{T}
    split::CoarseFineSplit
    F::T
    rcond::Float64
end

function FineBlockLU(A::SparseMatrixCSC, s::CoarseFineSplit; rcond_min::Real=1e-12)
    size(A) == (length(s), length(s)) || throw(DimensionMismatch("A is $(size(A)), split has $(length(s)) unknowns"))
    isempty(s.jf) && return FineBlockLU(s, nothing, 1.0)
    Aff = A[s.jf, s.jf]
    F = lu(Aff; check=false)
    rc = issuccess(F) ? rcond_estimate(Aff, F) : 0.0
    (isfinite(rc) && rc >= rcond_min) || throw(IllConditionedFineBlock(isfinite(rc) ? rc : 0.0, rcond_min))
    return FineBlockLU(s, F, rc)
end
FineBlockLU(S::NinePointStencil, s::CoarseFineSplit; kw...) = FineBlockLU(assemble_sparse(S), s; kw...)

"`x_f = A_ff⁻¹ b_f`."
fine_solve!(x, fb::FineBlockLU, b) = isempty(fb.split.jf) ? x : (x .= fb.F \ b; x)
"`x_f = A_ff⁻* b_f`."
fine_solve_adj!(x, fb::FineBlockLU, b) = isempty(fb.split.jf) ? x : (x .= fb.F' \ b; x)

"`A_αβ = P_α A E_β` of a sparse matrix (`α, β ∈ (:c, :f)`)."
block_matrix(A::AbstractMatrix, s::CoarseFineSplit, α::Symbol, β::Symbol) =
    A[_block_indices(s, α), _block_indices(s, β)]

function _inject(s::CoarseFineSplit, β::Symbol, v::AbstractVector)
    t = zeros(ComplexF64, length(s))
    t[_block_indices(s, β)] .= v
    return t
end

"""
    apply_block!(out, S, s, α, β, v)

Matrix-free block action `out = A_αβ v = P_α A (E_β v)` (spec Sec. 8): injection (zeros in
all other entries), one global stencil action, restriction.
"""
function apply_block!(out::AbstractVector, S::NinePointStencil, s::CoarseFineSplit, α::Symbol, β::Symbol, v::AbstractVector)
    w = apply_A!(zeros(ComplexF64, length(s)), S, _inject(s, β, v))
    out .= w[_block_indices(s, α)]
    return out
end
