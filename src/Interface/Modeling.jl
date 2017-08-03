export ForwModel

function ForwModel(v,Q,model,opts)
    Null = Array{Float64}(0)
    PDEfunc!(forw_model,vec(v),vec(Q),Null,Null,model,opts)
end
