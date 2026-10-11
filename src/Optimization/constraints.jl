export project_bounds!

project_bounds!(v::Vector{T},minv::Union{Vector{T},T},maxv::Union{Vector{T},T}) where {T<:Real} = project_bounds!(v,minv,maxv,T[])
    
function project_bounds!(v::Vector{T},minv::Union{Vector{T},T},maxv::Union{Vector{T},T},vproj::Vector{T}) where {T<:Real}    
    if length(vproj)==0
        vproj = copy(v)
    else
        vproj .= v
    end    
    @. vproj[vproj < minv] = minv
    @. vproj[vproj > maxv] = maxv
    return vproj
end
