export MLGMRES

function MLGMRES{I<:Integer,F<:AbstractFloat}(H::joAbstractOperator,v::AbstractArray{F,1},comp_grid::ComputationalGrid{I,F},model::Model{I,F},freq::Union{F,Complex{F}},opts::PDEopts{I,F})
    ks_outer = 3
    ks_inner = 5
    kc_outer = 3
    kc_inner = 5
    coarse_tol = 0.5
    
    nlevels = 3
    smoother = LinSolveOpts(solver=:fgmres,precond=:identity,maxit=ks_outer,maxinnerit=ks_inner)
    coarse_solver = LinSolveOpts(solver=:fgmres,precond=:identity,maxit=kc_outer,maxinnerit=kc_inner,tol=coarse_tol)
    (Hs,S,R,P,C) = construct_helm_multigrid(H,v,comp_grid,model,freq,opts,smoother,coarse_solver,nlevels)
    T = joMultigrid(Hs,S,R,P,C,smoother,coarse_solver,recursive_vcycle=true)    
end
