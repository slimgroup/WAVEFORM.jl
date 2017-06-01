export linearsolve, LinSolver, Precond, LinSolveOpts

const LinSolver = Set([:fgmres])
const Precond = Set([:mlgmres,:identity])

type LinSolveOpts
    tol::Float64
    maxit::Int64
    maxinnerit::Int64
    solver::Symbol
    precond::Union{Symbol,Function,LinSolveOpts,joAbstractOperator}
    outputfreq::Int64
    function LinSolveOpts(;tol::Float64=1e-6,maxit::Int64=10000,maxinnerit::Int64=20,solver::Symbol=:fgmres,precond::Union{Symbol,Function}=:identity,outputfreq::Int64=0)
        solver in LinSolver || throw(ArgumentError("Invalid solver $solver"))
        if typeof(precond)==Symbol
            precond in Precond || throw(ArgumentError("Invalid preconditioner: $precond"))
        end
        new(tol,maxit,maxinnerit,solver,precond,outputfreq)
    end
end

# Generic linear solution interface
#
function linearsolve(op,b,x0,forw_mode::Bool,lsopts::LinSolveOpts)
    if length(x0)==0
        x0 = zeros(b)
    end
    size(op,1)==size(op,2) || throw(ArgumentError("A must be square"))
    size(b,1)==size(op,1) || throw(ArgumentError("right hand side dimension mismatch"))
    size(x0,1)==size(op,1) || throw(ArgumentError("x0 dimension mismatch"))
    lsopts = deepcopy(lsopts)
    if forw_mode
        A = op
    else
        A = op'
    end
    if typeof(lsopts.precond)==LinSolveOpts
        prec_opts = lsopts.precond
        
        if prec_opts.solver==:fgmres
            P = x->FGMRES(A,x,zeros(x),m=prec_opts.maxinnerit,maxit=prec_opts.maxit,tol=prec_opts.tol)
        end
    elseif typeof(lsopts.precond)==Function        
        P = x->lsopts.precond(x,forw_mode)
    elseif typeof(lsopts.precond)==Symbol        
        if lsopts.precond==:identity
            P = nothing
        else
            throw(ArgumentError("Unrecognized preconditioner $lsopts.precond"))
        end
    elseif typeof(lsopts.precond)==joAbstractOperator
        if forw_mode
            P = x->lsopts.precond*x
        else
            P = x->lsopts.precond'*x
        end
    end

    if lsopts.solver==:fgmres
        (y,res) = FGMRES(A,b,x0,m=lsopts.maxinnerit,maxiter=lsopts.maxit,tol=lsopts.tol,precond=P,outputfreq=lsopts.outputfreq)
    else
        throw(ArgumentError("Unrecognized solver $lsopts.solver"))
    end
end
