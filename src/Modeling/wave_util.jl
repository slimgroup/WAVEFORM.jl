export odn_to_grid,grid_to_odn,fwi_wavelet,index_block

function index_block(idx,block_size::I) where {I<:Integer}
    m = div(length(idx),block_size)
    out_idx = [(i+1:i+block_size) for i in (0:m-1)*block_size]
    if length(idx) > m*block_size
        push!(out_idx, m*block_size+1:length(idx))
    end
    return [idx[out_idx[i]] for i in 1:length(out_idx)]
end

function fwi_wavelet(freqs::AbstractArray{F,1},t0::F,f0::F) where {F<:Real}
    w = exp.(-2*π*im*t0*freqs)
    if f0 > 0
        @. w = freqs^2 * exp(-(freqs./f0)^2)*w
    end
    return w
end

# Convert odn coordinates to their full grid coordinates
function odn_to_grid(o::AbstractArray{F,1},d::AbstractArray{F,1},n::AbstractArray{I,1}) where {F<:AbstractFloat, I<:Integer}
    (length(o)==length(d) && length(o)==length(n)) || throw(Exception("o,d,n must have the same length"))
    x = Array{StepRangeLen,1}(length(o))
    for i in 1:length(o)
        x[i] = o[i] + (0:(n[i]-1))*d[i]
    end
    return tuple(x...)
end


# Convert grid coordinates to odn coordinates
function grid_to_odn(x::AbstractArray{StepRangeLen{F},1}) where {F<:AbstractFloat}
    o = Array{F,1}()
    d = Array{F,1}()
    n = Array{Int64,1}()
    for i in 1:length(o)
        o[i] = x[i][1]
        d[i] = x[i][2] - x[i][1]
        n[i] = length(x[i])
    end
    return (o,d,n)
end
