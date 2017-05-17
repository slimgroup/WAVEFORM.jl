function PDEfunc{F<:AbstractFloat,I<:Integer}(op::PDEFUNC_OP,
                                              v::Union{AbstractArray{F,1},AbstractArray{Complex{F},1}},
                                              Q::Union{AbstractArray{F,1},AbstractArray{Complex{F},1}},
                                              Dobs::Union{AbstractArray{F,1},AbstractArray{Complex{F},1}},
                                              input::Union{AbstractArray{F,1},AbstractArray{Complex{F},1}},
                                              model::Model{I,F},
                                              opts::PDEopts{I,F};
                                              compute_grad::Bool=false,
                                              srcfreqmask::AbstractArray{Bool,2}=Array(Bool,0,0))

    nsrc = length(model.xsrc)*length(model.ysrc)*length(model.zsrc)
    nfreq = length(model.freq)
    nrec = length(model.xrec)*length(model.yrec)*length(model.zrec)
    nmodel = prod(model.n)
    if isempty(srcfreqmask)
        srcfreqmask = trues(nsrc,nfreq)
    end
    size(srcfreqmask,1)==nsrc && size(srcfreqmask,2)==nfreq || throw(ArgumentException("srcfreqmask must be a nsrc x nfreq matrix"))
    Iactive = find(vec(srcfreqmask))
    npde_out = length(Iactive)
    (iS,iF) = ind2sub((nsrc,nfreq),Iactive)
    freqsxsy = Array{Array{I,1},1}(nfreq)
    
    numcompsrc = (length(model.n)==2||model.n[3]==1) ? nsrc : 1
    length(Dobs)==0 || length(Dobs)==nrec*npde_out || throw(ArgumentException("Dobs must be a nrec*npde length vector"))
    
    if !isempty(Dobs)
        Dobs = reshape(Dobs,nrec,npde_out)
    end

    if length(Q)==nsrc*nsrc
        Q = reshape(Q,(nsrc,nsrc))
    elseif length(Q) ==nsrc*nsrc*nfreq
        Q = reshape(Q,(nsrc,nsrc,nfreq))
    else
        throw(ArgumentException("Q must be nsrc x nsrc or nsrc x nsrc x nfreq"))
    end
    for i in 1:nfreq
        J = find(iF.==i)
        if !isempty(J)
            freqsxsy[i] = iS[J]
        end
    end
    
    # No work to do, just return the right outputs 
    if npde_out==0
        z = zeros(eltype(v),(nmodel))
        if op==objective
            if compute_grad
                return (0.0,z)
            else
                return (0.0)
            end
            
        elseif op==forw_model || op==jacob_forw
            return (zeros(Complex{F},(nrec,0)))
        else
            return (z)
        end
    end
    
    # Set up outputs 
    if op==objective
        f = 0.0
        if compute_grad
            g = zeros(eltype(v),(nmodel))
        end
        misfit = opts.misfit
    elseif op==forw_model||op==jacob_forw
        output = zeros(Complex{F},(nrec,npde_out))
    else
        output = zeros(F,(nmodel))
        if op==jacob_adj
            input = reshape(input,nrec,npde_out)
        end
    end    
    npdes = 1
    freqs = model.freq
    w = fwi_wavelet(freqs,model.t0,model.f0)
    
    freq_idx = 0
    for k in 1:nfreq
        if isempty(freqsxsy[k])
            continue
        end
        freq_idx = freq_idx+1
        freq = freqs[k]
        isrc = freqsxsy[k]
        
        src_batches = index_block(isrc,numcompsrc)
        
        (H,comp_grid,T_forw,T_adj,DT_adj) = discrete_helmholtz(v,model,freq,opts)
        phys_to_comp = comp_grid.phys_to_comp_grid       
        comp_to_phys = comp_grid.comp_to_phys_grid
        (Ps,Pr) = src_rec_interp_operators(model,comp_grid)
        if length(find(op.==[jacob_forw hess_gn hess]))>0
            δm = phys_to_comp*input;
        end

        for j in 1:length(src_batches)
            current_src_idx = src_batches[j]
            # Scaling so that the wavefield amplitudes are the same for different grid spacings
            if ndims(Q)==3
                q = Q[:,current_src_idx,k]
            else
                q = Q[:,current_src_idx]
            end
            
            q = jo_convert(Complex{F},Ps*(w[k]*q))
            q = q*prod(model.d)/prod(comp_grid.comp_d)            
            
            data_idx = npdes:npdes+length(current_src_idx)-1
            u = H\q
            sum_srcs = x->comp_to_phys*sum(real(x),2)
            if op==objective
                (ϕ,δϕ) = misfit(Pr*u,Dobs[:,data_idx])
                f = f + ϕ
                if compute_grad
                    v = H'\(-(Pr'*δϕ))
                    g = g + sum_srcs(T_adj(u,v))
                end
            elseif op==field
                output[:,data_idx] = u
            elseif op==forw_model
                output[:,data_idx] = Pr*u            
            elseif op==jacob_forw
                δu = H\(-T_forw(u,δm))
                output[:,data_idx] = Pr*δu
            elseif op==jacob_adj
                v = H'\(-(Pr'*input[:,data_idx]))
                output = output + sum_srcs(T_adj(u,v))            
            elseif op==hess_gn
                δu = H\(-T_forw(u,δm))
                δu = H'\(-(Pr'*(Pr*δu)))
                output = output + sum_srcs(T_adj(u,δu))      
            elseif op==hess
                (ϕ,δϕ,δ2ϕ) = misfit(Pr*u,Dobs[:,data_idx])
                δu = H\(-T_forw(u,δm))
                v = H'\(-Pr'*δϕ)
                δv = H'\(-T_forw(v,δm) - Pr'*(δ2ϕ.* (Pr*δu) ) )
                output = output + sum_srcs(DT_adj(u,δm,δu)*v+T_adj(u,δv))
            end
            npdes = npdes + length(current_src_idx)
        end
    end
    if(op==objective)
        if compute_grad
            return (f,g)
        else
            return f
        end
    else
        return output
    end
end


function src_rec_interp_operators{F<:AbstractFloat,I<:Integer}(model::Model{I,F},comp_grid::ComputationalGrid{I,F})
    ndims = (length(model.n)==2||model.n[3]==1) ? 2 : 3
    kaiser_window_param = 4
    DT = Complex{F}
    RT = Complex{F}
    opS(x,y) = joSincInterp(x,y,DomainT=DT,RangeT=RT)
    if ndims==2        
        (zt,xt) = odn_to_grid(comp_grid)        
        Ps = joKron(opS(model.xsrc,xt),opS(model.zsrc,zt))
        Pr = joKron(opS(xt,model.xrec),opS(zt,model.zrec))
    else
        (xt,yt,zt) = odn_to_grid(comp_grid)
        Ps = joKron(opS(model.zsrc,zt),opS(model.ysrc,yt),opS(model.xsrc,xt))
        Pr = joKron(opS(zt,model.zrec),opS(yt,model.yrec),opS(xt,model.xrec))
    end
    return (Ps,Pr)
end
