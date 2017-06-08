export joMultigrid

function joMultigrid(Hs,S,R,P,C,coarse_solver;
                     recursive_vcycle::Bool=false)
    n = size(Hs[1],1)
    nlevels = length(Hs)
    if recursive_vcycle
        coarse_solves = Array{Function,1}()
        prepend!(coarse_solves,[solvesystem(Hs[nlevels],coarse_solver)])
        for i=nlevels-1:-1:1
            old_cs = coarse_solves[1]
            Hc = Hs[i+1]
            Hf = Hs[i]
            Rf = R[i]
            Pf = P[i]
            Sf = S[i]
            coarse_solver = deepcopy(coarse_solver)
            coarse_solver.precond = old_cs
            CS = solvesystem(Hc,coarse_solver)
            prepend!(coarse_solves,[(b,x,mode::Bool)->Waveform.mg_vcycle(Hf,b,x,Rf,Pf,CS,Sf,Sf,forw_mode=mode)])
        end
        solver = coarse_solves[1]
        return joLinearFunctionFwdCT(n,n,
                                     x->solver(x,zeros(x),true),
                                     x->solver(x,zeros(x),false),
                                     eltype(Hs[1]),name="Recursive Multigrid Preconditioner")
    else
        return joLinearFunctionFwdCT(n,n,
                             x->multigrid_multiply(Hs,S,R,P,C,coarse_solver,x,forw_mode=true),
                             x->multigrid_multiply(Hs,S,R,P,C,coarse_solver,x,forw_mode=false),
                             eltype(Hs[1]),name="V cycle Multigrid preconditioner")
    end

end

function multigrid_multiply(Hs,S,R,P,C,coarse_solver,b;forw_mode::Bool=true)
    x = zeros(b)
    x_lvl = Array{typeof(x),1}()
    b_lvl = Array{typeof(x),1}()
    push!(x_lvl,x)
    push!(b_lvl,b)
    nlevels = length(Hs)
    for i=1:nlevels-1
        xf = x_lvl[i]
        bf = b_lvl[i]
        xf = S[i](bf,xf,forw_mode)
        if forw_mode
            rf = bf-H[i]*xf
        else
            rf = bf-H[i]'*xf
        end
        x_lvl[i+1] = R[i]*rf
        b_lvl[i+1] = R[i]*bf
    end
    x = x_lvl[end]
    x = coarse_solver(xc,zeros(xc),forw_mode)
    for i=nlevels-1:-1:1
        xf = x_lvl[i] + Pr*x
        x = S[i](b_lvl[i],x,forw_mode)
    end
    return x

end
