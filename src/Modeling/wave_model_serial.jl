export PDEfunc!, pdefunc_ops
pdefunc_ops = [:objective,:field,:forw_model,:jacob_forw,:jacob_adj,:hess,:hess_gn]

function PDEfunc!(op::Symbol,
                  v::Union{AbstractArray{F,1},AbstractArray{Complex{F},1}},
                  Q::Union{AbstractArray{F,1},AbstractArray{Complex{F},1}},
                  Dobs::Union{AbstractArray{F,1},AbstractArray{Complex{F},1}},
                  input::Union{AbstractArray{F,1},AbstractArray{Complex{F},1}},
                  model::Model{I,F},
                  opts::PDEopts{I,F};
                  grad::Union{AbstractArray{F,1},AbstractArray{Complex{F},1}}=F[],
                  srcfreqmask::AbstractArray{Bool,2}=Matrix{Bool}(undef,0,0)) where {F<:AbstractFloat,I<:Integer}

    op in pdefunc_ops || error("Unrecognized op $op, must be one of $pdefunc_ops")
    nsrc = length(model.xsrc)*length(model.ysrc)*length(model.zsrc)
    nfreq = length(model.freq)
    nrec = length(model.xrec)*length(model.yrec)*length(model.zrec)
    nmodel = prod(model.n)
    if isempty(srcfreqmask)
        srcfreqmask = trues(nsrc,nfreq)
    end
    size(srcfreqmask,1)==nsrc && size(srcfreqmask,2)==nfreq || error("srcfreqmask must be a nsrc x nfreq matrix")
    Iactive = findall(vec(srcfreqmask))
    npde_out = length(Iactive)
    iSF = CartesianIndices((nsrc,nfreq))[Iactive]
    iS = getindex.(iSF,1)
    iF = getindex.(iSF,2)
    freqsxsy = Array{Array{I,1},1}(undef,nfreq)

    numcompsrc = (length(model.n)==2||model.n[3]==1) ? nsrc : 1
    length(Dobs)==0 || length(Dobs)==nrec*npde_out || error("Dobs must be a nrec*npde_out length vector")

    if !isempty(Dobs)
        Dobs = reshape(Dobs,nrec,npde_out)
    end

    if length(Q)==nsrc*nsrc
        Q = reshape(Q,(nsrc,nsrc))
    elseif length(Q) ==nsrc*nsrc*nfreq
        Q = reshape(Q,(nsrc,nsrc,nfreq))
    else
        throw(ArgumentError("Q must be nsrc x nsrc or nsrc x nsrc x nfreq"))
    end
    for i in 1:nfreq
        J = findall(iF.==i)
        if !isempty(J)
            freqsxsy[i] = iS[J]
        else
            freqsxsy[i] = Array{I,1}()
        end
    end

    # No work to do, just return the right outputs
    if npde_out==0
        z = zeros(eltype(v),(nmodel))
        if op==:objective
            return 0.0
        elseif op in [:forw_model,:jacob_forw]
            return (zeros(Complex{F},nrec,0))
        else
            return (z)
        end
    end

    # Set up outputs
    if op==:objective
        f = 0.0
        compute_grad = length(grad) > 0
        if compute_grad
            length(grad)==length(v) || error("length(grad) $(length(grad)) != length(v) $(length(v))")
            fill!(grad,0.0)
        end
        misfit = opts.misfit
    elseif op in [:forw_model,:jacob_forw]
        output = zeros(Complex{F},(nrec,npde_out))
    else
        output = zeros(F,(nmodel))
        if op==:jacob_adj
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

        (H,comp_grid,T,DT_adj) = helmholtz_system(v,model,freq,opts)
        phys_to_comp = comp_grid.phys_to_comp_grid
        comp_to_phys = comp_grid.comp_to_phys_grid
        (Ps,Pr) = src_rec_interp_operators(model,comp_grid)
        if op in [:jacob_forw :hess_gn :hess]
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

            # Wavefield solve
            U = H\q
            sum_srcs = x->real(comp_to_phys*vec(sum(x,dims=2)))
            if op==:objective
                #figure()
                #imshow(real(Pr*U),aspect="auto")
                #title("Pred data at $(model.freq[k]) Hz")
                (ϕ,δϕ) = misfit(Pr*U,Dobs[:,data_idx])
                f += ϕ
                if compute_grad
                    V = H'\(-(Pr'*δϕ))
                    grad .+= sum_srcs(T(U)'*V)
                end
            elseif op==:field
                output[:,data_idx] = U
            elseif op==:forw_model
                output[:,data_idx] = Pr*U
            elseif op==:jacob_forw
                δU = H\(-T(U)*δm)
                output[:,data_idx] = Pr*δU
            elseif op==:jacob_adj
                V = H'\(-(Pr'*input[:,data_idx]))
                output .+= sum_srcs(T(U)'*V)
            elseif op==:hess_gn
                δU = H\(-T(U)*δm)
                δU .= H'\(-(Pr'*(Pr*δU)))
                output .+= sum_srcs(T(U)*δU)
            elseif op==:hess
                (ϕ,δϕ,δ2ϕ) = misfit(Pr*U,Dobs[:,data_idx])
                δU = H\(-T(U)*δm)
                V = H'\(-Pr'*δϕ)
                δV = H'\(-T(V)*δm - Pr'*(δ2ϕ.* (Pr*δU) ) )
                output .+= sum_srcs(DT_adj(U,δm,δU)*V+T(U)*δV)
            end
            npdes += length(current_src_idx)
        end
    end
if(op==:objective)
    return f
else
    return output
end
end


function src_rec_interp_operators(model::Model{I,F},comp_grid::ComputationalGrid{I,F}) where {F<:AbstractFloat,I<:Integer}
    ndims = (length(model.n)==2||model.n[3]==1) ? 2 : 3
    DT = Complex{F}
    opS(x,y,h) = grid_sinc_interp(x,y,h,DT)
    if ndims==2
        (zt,xt) = odn_to_grid(comp_grid)
        (dz,dx) = comp_grid.comp_d
        Ps = joKron(opS(model.xsrc,xt,dx),opS(model.zsrc,zt,dz))
        Pr = joKron(opS(xt,model.xrec,dx),opS(zt,model.zrec,dz))
    else
        (xt,yt,zt) = odn_to_grid(comp_grid)
        (dx,dy,dz) = comp_grid.comp_d
        Ps = joKron(opS(model.zsrc,zt,dz),opS(model.ysrc,yt,dy),opS(model.xsrc,xt,dx))
        Pr = joKron(opS(zt,model.zrec,dz),opS(yt,model.yrec,dy),opS(xt,model.xrec,dx))
    end
    return (Ps,Pr)
end

# Sinc interpolation from grid xin to grid xout, with the sinc width set by
# the computational grid spacing h. (JOLI's joSincInterp infers the spacing
# from xin/xout, which fails when either is a single point.)
function grid_sinc_interp(xin::AbstractArray{F,1},xout::AbstractArray{F,1},h::F,DT::DataType) where {F<:AbstractFloat}
    return joMatrix(sinc.((xout .- xin')./h),DDT=DT,RDT=DT,name="grid_sinc_interp")
end
