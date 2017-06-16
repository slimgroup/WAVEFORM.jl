export construct_helm_multigrid

function construct_helm_multigrid{I<:Integer,F<:AbstractFloat}(H,v::AbstractArray{F,1},
        comp_grid::ComputationalGrid{I,F},
        model::Model{I,F},freq::Union{F,Complex{F}},
        opts::PDEopts{I,F},
        smoother::LinSolveOpts,
        coarse_solver::LinSolveOpts,nlevels::I; explicit_coarse_mat::Bool=false)

    coarse_factor = 2
    Hs = Array{ANY,1}()
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
        newopts.npml = div(pml_fine,coarse_factor^i)
        push!(S,solvesystem(H,smoother))
        (to_coarse,to_fine,ncoarse) = fine2coarse(nt_nopml_fine,div(newopts.comp_d,coarse_factor),newopts.comp_d,eltype(v))
        vcoarse = to_coarse*vec(v)
        newopts.comp_n = ncoarse
        if i==nlevels-1 && explicit_coarse_mat
            newopts.implicit_matrix = false
        end
        (H,comp_grid_coarse) = helmholtz_system(vcoarse,model,freq,newopts)
        (to_coarse,to_fine,ntcoarse) = fine2coarse(nt_fine,comp_grid_coarse.comp_n,eltype(H))
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
