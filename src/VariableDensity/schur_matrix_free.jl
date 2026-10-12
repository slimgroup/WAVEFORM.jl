export apply_schur!, apply_schur_adj!, reduced_rhs!, reconstruct_field

"""
    apply_schur!(out, S, s, fb, v_c)

`out = S v_c = A_cc v_c - A_cf A_ff⁻¹ A_fc v_c` with two global actions and one fine solve
(spec Sec. 8 pseudocode). The inverse is the solve held by `fb`.
"""
function apply_schur!(out::AbstractVector, S::NinePointStencil, s::CoarseFineSplit, fb::FineBlockLU, v::AbstractVector)
    w = apply_A!(zeros(ComplexF64, length(s)), S, _inject(s, :c, v))
    yf = fine_solve!(zeros(ComplexF64, length(s.jf)), fb, w[s.jf])
    corr = apply_A!(zeros(ComplexF64, length(s)), S, _inject(s, :f, yf))
    out .= (w .- corr)[s.ic]
    return out
end

"""
    apply_schur_adj!(out, S, s, fb, w_c)

`out = S* w_c = A_cc* w_c - A_fc* A_ff⁻* A_cf* w_c` from the adjoint stencil action and the
adjoint fine solve (spec Sec. 8), not from a conjugated Schur stencil.
"""
function apply_schur_adj!(out::AbstractVector, S::NinePointStencil, s::CoarseFineSplit, fb::FineBlockLU, v::AbstractVector)
    w = apply_Aadj!(zeros(ComplexF64, length(s)), S, _inject(s, :c, v))
    yf = fine_solve_adj!(zeros(ComplexF64, length(s.jf)), fb, w[s.jf])
    corr = apply_Aadj!(zeros(ComplexF64, length(s)), S, _inject(s, :f, yf))
    out .= (w .- corr)[s.ic]
    return out
end

"""
    reduced_rhs!(out_c, S, s, fb, f)

Reduced forcing `f_c - A_cf A_ff⁻¹ f_f` for a full-grid RHS `f` (for `f = B q` this is
`B_eff q`).
"""
function reduced_rhs!(out::AbstractVector, S::NinePointStencil, s::CoarseFineSplit, fb::FineBlockLU, f::AbstractVector)
    yf = fine_solve!(zeros(ComplexF64, length(s.jf)), fb, f[s.jf])
    corr = apply_A!(zeros(ComplexF64, length(s)), S, _inject(s, :f, yf))
    out .= f[s.ic] .- corr[s.ic]
    return out
end

"""
    reconstruct_field(S, s, fb, f, u_c)

Full field from retained values: `u_f = A_ff⁻¹ (f_f - A_fc u_c)`, `u = E_c u_c + E_f u_f`.
"""
function reconstruct_field(S::NinePointStencil, s::CoarseFineSplit, fb::FineBlockLU, f::AbstractVector, uc::AbstractVector)
    afc = apply_A!(zeros(ComplexF64, length(s)), S, _inject(s, :c, uc))
    uf = fine_solve!(zeros(ComplexF64, length(s.jf)), fb, f[s.jf] .- afc[s.jf])
    u = _inject(s, :c, uc)
    u[s.jf] .= uf
    return u
end
