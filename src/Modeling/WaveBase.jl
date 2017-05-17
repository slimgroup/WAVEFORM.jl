export Model, PDEFUNC_OP, STENCIL2D, STENCIL3D, PDEopts

type Model{IntType<:Integer,FloatType<:AbstractFloat}
    n::AbstractArray{IntType,1}
    d::AbstractArray{FloatType,1}
    o::AbstractArray{FloatType,1}
    t0::FloatType
    f0::FloatType
    unit::String
    freq::AbstractArray{FloatType,1}
    xsrc::AbstractArray{FloatType,1}
    ysrc::AbstractArray{FloatType,1}
    zsrc::AbstractArray{FloatType,1}
    xrec::AbstractArray{FloatType,1}
    yrec::AbstractArray{FloatType,1}
    zrec::AbstractArray{FloatType,1}
end



@enum PDEFUNC_OP objective field forw_model jacob_forw jacob_adj hess hess_gn
@enum STENCIL2D helm2d_chen9p helm2d_std7
@enum STENCIL3D helm3d_operto27 helm3d_std9


type PDEopts{IntType<:Integer,FloatType<:AbstractFloat}
    pde_scheme::Union{STENCIL2D,STENCIL3D}
    comp_n::AbstractArray{IntType,1}
    comp_d::AbstractArray{FloatType,1}
    comp_o::AbstractArray{FloatType,1}
    cut_pml::Bool
    npml::AbstractArray{IntType,2}
    misfit::Function
end


