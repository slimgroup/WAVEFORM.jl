function mg_vcycle(A,b,x,R,P,coarse_solve,S_pre,S_post=S_pre; maxit::Integer=1,forw_mode::Bool=true)
    if !forw_mode
        A = A'
    end
    if length(x)==0
        x = zeros(b)
    end
    for i=1:maxit
        x = S_pre(b,x,forw_mode)
        r = b - A*x
        xc = R*r
        xc = coarse_solve(xc,zeros(xc),forw_mode)
        x = x + P*xc
        x = S_post(b,x,forw_mode)
    end
    return x
end
