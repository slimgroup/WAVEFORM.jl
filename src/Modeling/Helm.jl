using JOLI

export discrete_helmholtz, ComputationalGrid, odn_to_grid

type ComputationalGrid{I<:Integer,F<:AbstractFloat}
    phys_to_comp_grid::joAbstractOperator
    comp_to_phys_grid::joAbstractOperator

    comp_o::AbstractArray{F,1}
    comp_d::AbstractArray{F,1}
    comp_n::AbstractArray{I,1}
    comp_n_nopml::AbstractArray{I,1}
end

function odn_to_grid{F<:AbstractFloat,I<:Integer}(comp_grid::ComputationalGrid{I,F})
    return odn_to_grid(comp_grid.comp_o,comp_grid.comp_d,comp_grid.comp_n)
end

function helmholtz_system{I<:Integer,F<:AbstractFloat}(v::AbstractArray{F,1},model::Model{I,F},freq::Union{F,Complex{F}},opts::PDEopts{I,F})
    
    ndims = (length(opts.comp_n)==2 || opts.comp_n[3]==1) ? 2 : 3
    lsopts = deepcopy(opts.lsopts)
    if ndims==2
        opts.comp_n = opts.comp_n[1:2]
        opts.comp_d = opts.comp_d[1:2]
        opts.comp_o = opts.comp_o[1:2]
    end
    v = reshape(v,tuple(opts.comp_n...))
    dt = opts.comp_d
    if model.unit == "m/s"
        vmax = maximum(vec(v))
    elseif model.unit == "s2/m2"
        vmax = maximum(vec(v).^(-1/2))
    elseif model.unit == "s2/km2"
        vmax = maximum(1e3*(vec(v).^(-1/2)))
    else
        throw(ArgumentError())
    end
    npml = convert(Array{I,1},ceil.(vmax./(real(freq)*dt)))
    npml = repmat(npml',2,1)
    npml = min(npml,opts.npml)
    # npml is a 2x3 matrix corresponding to
    # in 2D
    # [ #pml pts lower z  | # pml pts lower x  | 0 ]
    # [ #pml pts upper z  | # pml pts upper x  | 0 ]
    
    # in 3D
    # [ #pml pts lower x  | # pml pts lower y  | # pml pts lower z ]
    # [ #pml pts upper x  | # pml pts upper y  | # pml pts upper z ]
    
    nt_nopml = opts.comp_n
    nt_pml = nt_nopml + sum(npml,1)'
    nt_pml = vec(nt_pml)
    ot_pml = opts.comp_o - vec(npml[1,:]) .* vec(dt)
    (Pext,Ppad) = get_pad_ext_ops(nt_nopml,npml,ndims)
    phys_to_comp = Pext
    if opts.cut_pml
        comp_to_phys = Ppad'
    else
        comp_to_phys = Pext'
    end
    v_pml = reshape(Pext*vec(v),tuple(nt_pml...))
    comp_grid = ComputationalGrid{I,F}(phys_to_comp,comp_to_phys,vec(ot_pml),vec(dt),vec(nt_pml),vec(nt_nopml))
    N_system = prod(nt_pml)

    # Set up preconditioner
    
    if ndims==2
        if opts.pde_scheme==helm2d_chen9p
            (H,dH,ddH) = helm2d_chen2013(nt_pml,dt,npml,freq,v_pml,model.f0,model.unit)
        elseif opts.pde_scheme==helm2d_std7
            (H,dH,ddH) = helm2d_std7(nt_pml,dt,npml,freq,v_pml,model.f0,model.unit)
        end
        if lsopts.solver==:lufact
            opH = joInvertibleMatrix(H)
        else
            throw(ArgumentError("Unimplemented method $(opts.lsopts.solver) for 2D"))
        end
    elseif ndims==3
        if opts.pde_scheme==helm3d_operto27
            (wn,dwn,ddwn) = param_to_wavenum(v_pml,freq,model.unit)
            if !opts.implicit_matrix
                H = helm3d_operto_matrix(wn,dt,nt_pml,freq,npml)
            else
                H = (x;forw_mode=true)->helm3d_operto_mvp(wn,dt,nt_pml,freq,npml,reshape(x,nt_pml...))
            end
            
        elseif opts.pde_scheme==helm3d_std9
            
        end
        if lsopts.precond==:mlgmres
            lsopts.precond = MLGMRES(H,v,comp_grid,model,freq,opts)
        else
            lsopts.precond = :identity
        end

        if lsopts.solver==:lufact
            opH = joInvertibleMatrix(H)
        else
            if opts.implicit_matrix
                opH = joLinearFunctionCT(N_system,N_system,
                                         x->H(x,forw_mode=true),
                                         x->H(x,forw_mode=false),
                                         x->linearsolve(H,x,[],lsopts,forw_mode=true),
                                         x->linearsolve(H,x,[],lsopts,forw_mode=false),
                                         Complex{F},Complex{F})
            else
                opH = H
            end
        end
    end

    
    T = u-> joLinearFunctionFwdT( N_system,N_system,
                                  dm->dH*(dm.*u),
                                  z->conj(u).*(dH'*z),Complex{F},fMVok=true)
    DTadj = (u,dm,du)->joLinearFunctionFwdT(prod(nt_pml),prod(nt_pml),
                                            z->conj(u).*(dm.*(ddH'*z)) + conj(du).*(dH'*z),
                                            @joNF,Complex{F},fMVok=true)

    return (opH,comp_grid,T,DTadj)
end



function get_pad_ext_ops(nt_nopml,npml,ndims)
    if ndims == 2
        Pext = joKron(joExtension(nt_nopml[2],JOLI.pad_border,pad_lower=npml[1,2],pad_upper=npml[2,2]),
                      joExtension(nt_nopml[1],JOLI.pad_border,pad_lower=npml[1,1],pad_upper=npml[2,1]))
        Ppad = joKron(joExtension(nt_nopml[2],JOLI.pad_zeros,pad_lower=npml[1,2],pad_upper=npml[2,2]),
                      joExtension(nt_nopml[1],JOLI.pad_zeros,pad_lower=npml[1,1],pad_upper=npml[2,1]))
    else

        Pext = joKron(joExtension(nt_nopml[3],JOLI.pad_border,pad_lower=npml[1,3],pad_upper=npml[2,3]),
                      joExtension(nt_nopml[2],JOLI.pad_border,pad_lower=npml[1,2],pad_upper=npml[2,2]),
                      joExtension(nt_nopml[1],JOLI.pad_border,pad_lower=npml[1,1],pad_upper=npml[2,1]))
        Ppad = joKron(joExtension(nt_nopml[3],JOLI.pad_zeros,pad_lower=npml[1,3],pad_upper=npml[2,3]),
                      joExtension(nt_nopml[2],JOLI.pad_zeros,pad_lower=npml[1,2],pad_upper=npml[2,2]),
                      joExtension(nt_nopml[1],JOLI.pad_zeros,pad_lower=npml[1,1],pad_upper=npml[2,1]))
    end
    return (Pext,Ppad)
end

