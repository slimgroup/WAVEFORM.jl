function MLGMRES{I<:Integer,F<:AbstractFloat}(H::joAbstractOperator,v::AbstractArray{F,1},comp_grid::ComputationalGrid{I,F},model::Model{I,F},freq::Union{F,Complex{F}},opts::PDEopts{I,F}) 

    smoother = Waveform.LinSolveOpts(solver=:fgmres,maxit=1,maxinnerit=5,precond=:identity);
    coarse_solver = Waveform.LinSolveOpts(solver=:fgmres,maxit=1,maxinnerit=5,tol=0.5);
    nlevels = 2;
    (Hs,S,R,P,C) = Waveform.construct_helm_multigrid(H,v,comp_grid,model,freq,opts,smoother,coarse_solver,nlevels,explicit_coarse_mat=false);
    M = Waveform.joMultigrid(Hs,S,R,P,C,coarse_solver,recursive_vcycle=false)
end
