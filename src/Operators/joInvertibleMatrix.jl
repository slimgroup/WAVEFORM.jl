function joInvertibleMatrix{F<:Number}(A::AbstractMatrix{F};DDT::DataType=F,RDT::DataType=promote_type(F,DDT))
    (m,n) = size(A)
    m==n || throw(ArgumentError("A must be square"))
    T = lufact(A);
    (L,U,p,q,R) = T[:(:)];
    P = joPermutation(p,DDT=DDT)
    Q = joPermutation(q,DDT=DDT)
    Ut = U'
    Lt = L'
    forw_div = v->Q'*(U\(L\(P*(R.*v))))
    adj_div = v->conj(R).*(P'*(Lt\(Ut\(Q*v))))
    
    return joLinearFunctionCT(n,n,
                               v->A*v,                               
                               v->A'*v,
                               forw_div,
                               adj_div,
                               DDT,RDT,name="joInvertibleMatrix",fMVok=true,iMVok=true)
end

