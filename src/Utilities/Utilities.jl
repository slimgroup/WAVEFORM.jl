export fine2coarse, partition

# Generates interpolators from a fine grid to a coarse grid
#
# Usage:
#   (f2c,c2f) = fine2coarse(n_fine,n_coarse,T)
#
# Inputs
#    n_fine   - 1, 2, or 3 dimensional vector of fine grid sizes
#    n_coarse - vector of coarse grid sizes (same dimensionality as n_fine)
#    T        - vector data type
#
# Outputs
#   f2c       - fine to coarse grid JOLI operator
#   c2f       - coarse to fine grid JOLI operator
#
# Usage:
#    (f2c,c2f,n_coarse) = fine2coarse(n_fine,d_fine,d_coarse,T)
#
# Inputs
#    n_fine   - 1, 2, or 3 dimensional vector of fine grid sizes
#    d_fine   - vector of fine grid spacings (same dimensionality as n_fine)
#    d_coarse - vector of coarse grid spacings (same dimensionality as d_fine)
#    T        - vector data type
#
# Outputs
#   f2c       - fine to coarse grid JOLI operator
#   c2f       - coarse to fine grid JOLI operator
#   n_coarse  - vector of coarse grid sizes

function fine2coarse(n,d,d_sub...;interp_type::Symbol=:linear)

    if length(d_sub)==1
        n_sub = d
        typeof(d_sub[1])==DataType || error("Third argument must be of type DataType")
        T = d_sub[1]
    elseif length(d_sub)==2
        n_sub = ceil.(Int64,n.*d./d_sub[1])
        typeof(d_sub[2])==DataType || error("Fourth argument must be of type DataType")
        T = d_sub[2]
    else
        throw(ArgumentError("Too many arguments"))
    end
    length(n)==length(n_sub) || throwArgumentError("n and n_sub must have the same length")
    minimum(n./n_sub)>=1 || throw(ArgumentError("n_sub must be smaller than n elementwise"))
    if interp_type==:linear
        interp_basis = (xin,xout)->joLinInterp1D(xin,xout,T)
    elseif interp_type==:cubic
        interp_basis = (xin,xout)->joLagrangeInterp1D(xin,xout,T)
    else
        error("interp_type $(interp_type) must be :linear or :cubic")
    end
    ndims = length(n)
    if ndims==1
        f2c = interp_basis(linspace(0,1,n),linspace(0,1,n_sub))
        c2f = interp_basis(linspace(0,1,n_sub),linspace(0,1,n))
    elseif ndims==2
        f2c = joKron(interp_basis(linspace(0,1,n[2]),linspace(0,1,n_sub[2])),
                     interp_basis(linspace(0,1,n[1]),linspace(0,1,n_sub[1])))
        c2f = joKron(interp_basis(linspace(0,1,n_sub[2]),linspace(0,1,n[2])),
                     interp_basis(linspace(0,1,n_sub[1]),linspace(0,1,n[1])))
    elseif ndims==3
        c2f = joKron(interp_basis(linspace(0,1,n_sub[3]),linspace(0,1,n[3])),
                     interp_basis(linspace(0,1,n_sub[2]),linspace(0,1,n[2])),
                     interp_basis(linspace(0,1,n_sub[1]),linspace(0,1,n[1])))
        if interp_type==:linear
            f2c = c2f'
        else
            f2c = joKron(interp_basis(linspace(0,1,n[3]),linspace(0,1,n_sub[3])),
                         interp_basis(linspace(0,1,n[2]),linspace(0,1,n_sub[2])),
                         interp_basis(linspace(0,1,n[1]),linspace(0,1,n_sub[1])))
        end

    else
        throw(ArgumentError("n must be 1, 2, or 3 dimensional"))
    end
    return (f2c,c2f,n_sub)
end

# Partition the vector 1:N in to chunks of size P with a given overlap
# Note, some combinations of N, P, overlap will not yield a partition that goes exactly
# up to N, so a row with a possibly different overlap could be added to the end
#
# Usage:
#   y = partition(N,P,overlap)
#
# Input:
#   N       - number of elements
#   P       - partition size
#   overlap - number of elements to overlap
#
# Output:
#   y       - matrix of indices of size P x # partitions
#             each column correponds to a chunk of indices
function partition(N,P,overlap)
    y = collect(1:P) .+ collect(0:(P-overlap):(N-P))'
    if maximum(y[:,end]) < N
        y = [y; (N-P+1):N]
    end    
    return y
end
