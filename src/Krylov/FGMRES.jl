function FGMRES{T<:Number}(A,
                           b::AbstractVector{T},
                           x0::AbstractVector{T};
                           m::Integer=10,
                           maxiter=Inf,
                           precond=nothing,
                           tol::AbstractFloat=1e-6,
                           outputfreq::Integer=0)
    n = size(A,2)

    size(A,1)==size(A,2) || throw(ArgumentError("A must be square"))
    size(b,1)==n || throw(ArgumentError("A and b must have compatible dimensions"))
    size(b,2)==1 || throw(ArgumentError("b must be a vector"))
    length(x0)==0 || length(x0)==n || throw(ArgumentErrror("x0 must be empty or a vector with size compatible with A"))
    if length(x0)==0 || norm(x0)<1e-10
        x = zeros(T,size(A,2))
        r = b
    else
        x = x0
        r = b - A*x
    end
    normr0 = norm(r)
    ej = zeros(T,m+1)
    ej[1] = 1
    if precond==nothing
        P = x->x
        prec_dirac = true
    elseif typeof(precond)<:Function
        P = precond
        prec_dirac = false
    elseif typeof(precond)<:joAbstractOperator
        P = x->precond*x
        prec_dirac = false
    end
    
    hst = Array{Float64,1}()
    push!(hst,1.0)
    res = Array{Float64,1}()
    push!(res,1.0)
    it_counter = 1

    if prec_dirac
        Z = zeros(T,n,m+1)
    else
        Z = zeros(T,n,m+1)
        V = zeros(T,n,m+1)
    end

    while hst[end] > tol && div(it_counter,m)<maxiter
        Z = 0*Z;
        if !prec_dirac
            V = 0*V;
        end
        H = zeros(T,m+1,m)
        y = zeros(T,m)
        beta = norm(r)
        if prec_dirac
            Z[:,1] = r/beta
        else
            V[:,1] = r/beta
            @show norm(V[:,1])
            return (V[:,1],res)
        end
        j = 1
        for k=1:m
            if !prec_dirac
                Z[:,k] = P(V[:,k])
            end
            w = A*Z[:,k]

            if prec_dirac
                H[1:k,k] = Z[:,1:k]'*w
                w = w-Z[:,1:k]*H[1:k,k]
            else
                H[1:k,k] = V[:,1:k]'*w
                w = w-V[:,1:k]*H[1:k,k]
            end
            H[k+1,k] = norm(w)
            if outputfreq > 0
                @show H[1:k+1,k]
            end
            if prec_dirac
                Z[:,k+1] = w./H[k+1,k]
            else
                V[:,k+1] = w./H[k+1,k]
            end

            y[1:k] = H[1:k+1,1:k]\(beta*ej[1:k+1])

            push!(hst,norm(beta*ej[1:k+1]-H[1:k+1,1:k]*y[1:k])/normr0)
            j+=1

            if hst[end] < tol
                break
            end
            it_counter += 1
        end
        x = x + Z[:,1:j-1]*y[1:j-1]
        r = b - A*x
        if outputfreq > 0 && mod(div(it_counter,m),outputfreq)==0
            @printf("it %3d res %3.3e\n",div(it_counter,m), norm(r)/normr0)
        end
        push!(res,norm(r))
        if norm(r)/normr0<tol
            break
        end
    end
    return (x,res)
end
