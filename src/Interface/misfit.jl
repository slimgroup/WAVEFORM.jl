export construct_pde_misfit

function construct_pde_misfit(v,Q,D,model,opts;batch_mode::Bool=false)
    nsrc = length(model.xsrc)*length(model.ysrc)*length(model.zsrc)
    nfreq = length(model.freq)
    sf_mask = findall(opts.srcfreqmask)

    if length(sf_mask) < nsrc*nfreq
        D1 = view(D,:,sf_mask)
    else
        D1 = D
    end
    if batch_mode
        return (I)->(v,g)->misfit_func!(v,Q,D1,model,opts,I,g)
    else
        return (v,g)->misfit_func!(vec(v),vec(Q),vec(D1),model,opts,Array{Int64}(undef,0),g)
    end
end

function misfit_func!(v,Q,D,model,opts,I,g)
    if length(I)==0
        I = 1:length(findall(opts.srcfreqmask))
    end
    Null = Array{Float64,1}(undef, 0)
    nsrc = length(model.xsrc)*length(model.ysrc)*length(model.zsrc)
    nfreq = length(model.freq)
    (size(opts.srcfreqmask,1)==nsrc && size(opts.srcfreqmask,2)==nfreq) || error("opts.srcfreqmask must be nsrc x nfreq, got $(size(opts.srcfreqmask,1)) x $(size(opts.srcfreqmask,2))")
    Ifixed = findall(opts.srcfreqmask)
    if length(Ifixed)==0
        Ifixed = 1:nsrc*nfreq
    end
    maximum(I) <= length(Ifixed) || error("Requested index set of size $(maximum(I)) out of range of total number of sources/frequencies $(length(Ifixed))")
    sfmask = falses(nsrc,nfreq)
    sfmask[Ifixed[I]] .= true
    if length(I)<length(Ifixed)
        Dobs = D[:,find(sfmask)]
    else
        Dobs = D
    end
    c = 1/length(I)
    f = PDEfunc!(:objective,v,Q,Dobs,Null,model,opts,grad=g,srcfreqmask=sfmask)
    f = c*f
    @. g = c*g
    return f
end
