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
            coarse_solver.tag = "CS lvl "*string(i)
            coarse_solver.precond = old_cs
            CS = solvesystem(Hc,coarse_solver)
            if i==1
                ret = :coarse
            else
                ret = :preS1
            end
            prepend!(coarse_solves,[(b,x,mode::Bool)->multigrid_vcycle(Hf,Sf,Rf,Pf,CS,b,x,forw_mode=mode,ret=ret)])
        end
        solver = coarse_solves[1]
        return joLinearFunctionFwdCT(n,n,
                                     x->solver(x,zeros(x),true),
                                     x->solver(x,zeros(x),false),
                                     eltype(Hs[1]),name="Recursive Multigrid Preconditioner")
    else
        return joLinearFunctionFwdCT(n,n,
                             x->multigrid_multiply(Hs,S,R,P,C,x,forw_mode=true),
                             x->multigrid_multiply(Hs,S,R,P,C,x,forw_mode=false),
                             eltype(Hs[1]),name="V cycle Multigrid preconditioner")
    end

end

function multigrid_vcycle(H,S,R,P,C,b,x;forw_mode::Bool=true,ret::Symbol==:none)
    if ret==:preS1
        return x
    end
    xf = S(b,x,forw_mode)
    if ret==:postS1
        return x
    end
    if forw_mode
        r = b-H*xf
    else
        r = b-H'*xf
    end
    if ret==:r
        return r
    end
    xc = R*r
    if ret==:xc
        return xc
    end
    xc = C(xc,zeros(xc),forw_mode)
    if ret==:coarse
        return xc
    end
    if ret==:Pxc
        return P*xc
    end
    xf .+= P*xc
    if ret==:preS2
        return xf
    end
    xf = S(b,xf,forw_mode)
end

function multigrid_multiply(Hs,S,R,P,C,b;forw_mode::Bool=true)
    x = zeros(b)
    x_lvl = Array{typeof(b),1}()
    b_lvl = Array{typeof(b),1}()
    push!(x_lvl,x)
    push!(b_lvl,b)
    nlevels = length(Hs)
    for i=1:nlevels-1
        xf = x_lvl[i]
        bf = b_lvl[i]
        xf = S[i](bf,xf,forw_mode)
        if forw_mode
            rf = bf-Hs[i]*xf
        else
            rf = bf-Hs[i]'*xf
        end
        push!(x_lvl,R[i]*rf)        
        push!(b_lvl,R[i]*bf)
    end
    x = x_lvl[end]
    x = C(x,zeros(x),forw_mode)
    for i=nlevels-1:-1:1
        xf = x_lvl[i] + P[i]*x
        x = S[i](b_lvl[i],xf,forw_mode)
    end
    return x

end
