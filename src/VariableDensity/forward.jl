export factorize_A, solve_A!, predict_data

"Sparse LU of the system matrix: the direct-solver oracle for small problems."
factorize_A(op::HelmholtzPML) = lu(assemble_sparse(op))

"""
    solve_A!(x, F, rhs)

`x = A⁻¹ rhs` using a factorization from `factorize_A` (spec Sec. 12 interface).
"""
solve_A!(x::AbstractVecOrMat, F, rhs::AbstractVecOrMat) = ldiv!(x, F, rhs)

"""
    predict_data(op, acq, q=I; snap=1e-9)

Receiver data `D = R A⁻¹ B q` (`nrec × nsrc`, or `nrec` for a vector `q`) for one material
realization and frequency; all sources share the same factorization.
"""
function predict_data(op::HelmholtzPML, acq::Acquisition, q=I; snap::Real=1e-9)
    g = op.grid
    B = source_matrix(g, acq, op.weights; snap)
    R = receiver_matrix(g, acq; snap)
    rhs = B * q
    U = factorize_A(op) \ (q isa AbstractVector ? Vector{ComplexF64}(rhs) : Matrix{ComplexF64}(rhs))
    return R * U
end
