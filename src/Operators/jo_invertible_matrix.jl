function joInvertibleMatrix(A::AbstractMatrix{F};DDT::DataType=F,RDT::DataType=promote_type(F,DDT)) where {F<:Number}
    (m,n) = size(A)
    m==n || throw(ArgumentError("A must be square"))
    T = lu(A)
    forw_div = v->T\v
    adj_div = v->T'\v

    return joLinearFunction_A(n,n,
                              v->A*v,                               
                              v->A'*v,
                              forw_div,
                              adj_div,
                              DDT,RDT,name="joInvertibleMatrix",fMVok=true,iMVok=true)
end

