export construct_helm_multigrid, VGMRES, MLGMRES


function construct_helm_multigrid(H,v::AbstractArray{F,1},
        comp_grid::ComputationalGrid{I,F},
        model::Model{I,F},freq::Union{F,Complex{F}},
        opts::PDEopts{I,F},
        smoother::LinSolveOpts,
        coarse_solver::LinSolveOpts,nlevels::I; explicit_coarse_mat::Bool=false) where {I<:Integer,F<:AbstractFloat}

    coarse_factor = 2
    Hs = Array{Any,1}()
    push!(Hs,H)
    S = Array{Function,1}()
    R = Array{joAbstractOperator,1}()
    P = Array{joAbstractOperator,1}()
    pml_fine = opts.npml
    nt_fine = comp_grid.comp_n
    dt_fine = comp_grid.comp_d
    nt_nopml_fine = comp_grid.comp_n_nopml
    c = 1/8
    for i=1:nlevels-1
        newopts = deepcopy(opts)
        newopts.lsopts.precond = :identity
        newopts.comp_d = coarse_factor^i*dt_fine
        @. newopts.npml = round(Int64,ceil(float(pml_fine)/coarse_factor^i))
        push!(S,solvesystem(Hs[i],smoother))
        (to_coarse,to_fine,ncoarse) = fine2coarse(nt_nopml_fine,newopts.comp_d/coarse_factor,newopts.comp_d,eltype(v),interp_type=:cubic)

        vcoarse = to_coarse*vec(v)
        newopts.comp_n = ncoarse
        if i==nlevels-1 && explicit_coarse_mat
            newopts.implicit_matrix = false
        end
        (H,comp_grid_coarse) = helmholtz_system(vcoarse,model,freq,newopts)
        (to_coarse,to_fine,ntcoarse) = fine2coarse(nt_fine,comp_grid_coarse.comp_n,eltype(H),interp_type=:linear)
        push!(R,c*to_coarse)
        push!(P,to_fine)
        push!(Hs,H)
        v = vcoarse
        nt_nopml_fine = ncoarse
        nt_fine = ntcoarse
    end
    C = solvesystem(Hs[end],coarse_solver)
    return (Hs,S,R,P,C)
end

function VGMRES(H::joAbstractOperator,v::AbstractArray{F,1},comp_grid::ComputationalGrid{I,F},model::Model{I,F},freq::Union{F,Complex{F}},opts::PDEopts{I,F}) where {I<:Integer,F<:AbstractFloat}
    smoother = LinSolveOpts(solver=:fgmres,maxit=1,maxinnerit=5,precond=:identity);
    coarse_solver = LinSolveOpts(solver=:fgmres,maxit=1,maxinnerit=5,tol=0.5);
    nlevels = 2;
    (Hs,S,R,P,C) = construct_helm_multigrid(H,v,comp_grid,model,freq,opts,smoother,coarse_solver,nlevels,explicit_coarse_mat=false);
    M = joMultigrid(Hs,S,R,P,C,coarse_solver,recursive_vcycle=false)
end

function MLGMRES(H::joAbstractOperator,v::AbstractArray{F,1},comp_grid::ComputationalGrid{I,F},model::Model{I,F},freq::Union{F,Complex{F}},opts::PDEopts{I,F}) where {I<:Integer,F<:AbstractFloat}
    smoother = LinSolveOpts(solver=:fgmres,maxit=3,maxinnerit=5,precond=:identity);
    coarse_solver = LinSolveOpts(solver=:fgmres,maxit=3,maxinnerit=5,tol=0.5);
    nlevels = 3;
    (Hs,S,R,P,C) = construct_helm_multigrid(H,v,comp_grid,model,freq,opts,smoother,coarse_solver,nlevels,explicit_coarse_mat=false);
    M = joMultigrid(Hs,S,R,P,C,coarse_solver,recursive_vcycle=true)
end
