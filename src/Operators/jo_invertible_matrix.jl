function joInvertibleMatrix(A::AbstractMatrix{F};DDT::DataType=F,RDT::DataType=promote_type(F,DDT)) where {F<:Number}
    (m,n) = size(A)
    m==n || throw(ArgumentError("A must be square"))
    T = lu(A);
    P = joPermutation(T.p,DDT=DDT)
    Q = joPermutation(T.q,DDT=DDT)
    Ut = T.U'
    Lt = T.L'
    forw_div = v->Q'*(T.U\(T.L\(P*(T.Rs .* v))))
    adj_div = v->conj(T.Rs) .* (P'*(Lt\(Ut\(Q*v))))
    
    return joLinearFunction_A(n,n,
                              v->A*v,                               
                              v->A'*v,
                              forw_div,
                              adj_div,
                              DDT,RDT,name="joInvertibleMatrix",fMVok=true,iMVok=true)
end

