struct helm3d_params{T}
    w3a::T
    wm2::T
    wm3::T
    wm4::T
    cNNN::T
    cx::T
    cy::T
    cz::T
    xy_coef::T
    xz_coef::T
    yz_coef::T
    px_lo::Array{Complex{T},1}
    px_hi::Array{Complex{T},1}
    px::Array{Complex{T},1}
    py_lo::Array{Complex{T},1}
    py_hi::Array{Complex{T},1}
    py::Array{Complex{T},1}
    pz_lo::Array{Complex{T},1}
    pz_hi::Array{Complex{T},1}
    pz::Array{Complex{T},1}
end

function helm3d_operto_mvp_forw_impl!(wn, x, y, n, params, deriv_mode)
    M, N, P = 1, 2, 3
    zero_x = complex(0.0,0.0)
    zero_w = zero(eltype(wn))
    nx,ny,nz = n
    is_deriv_mode = deriv_mode ? complex(0.0,0.0) : complex(1.0,0.0)
    num_blocks = Threads.nthreads()
    block_size = ceil(Int64,nz/num_blocks)
    Threads.@threads for i=1:num_blocks
        z_idx = (i-1)*block_size+1:min(nz,i*block_size)
        wn_window = zeros(eltype(wn),3,3,3)
        coef = zeros(Complex{Float64},3,3,3)
        x_window = zeros(Complex{Float64},3,3,3)

        for k = z_idx
            cxz = (params.xz_coef*params.pz[k])
            cxzlo = (params.xz_coef*params.pz_lo[k])
            cxzhi = (params.xz_coef*params.pz_hi[k])
            cyz = (params.yz_coef*params.pz[k])
            cyzlo = (params.yz_coef*params.pz_lo[k])
            cyzhi = (params.yz_coef*params.pz_hi[k])
            for j = 1:ny
                c1 = params.xy_coef*params.py[j] + cxz
                c2_lo = params.cy*params.py_lo[j] + cyz
                c2_hi = params.cy*params.py_hi[j] + cyz
                c3_lo = params.yz_coef*params.py[j] + params.cz*params.pz_lo[k]
                c3_hi = params.yz_coef*params.py[j] + params.cz*params.pz_hi[k]

                dyzLL = - params.yz_coef*params.py_lo[j] - params.yz_coef*params.pz_lo[k]
                dyzLH = - params.yz_coef*params.py_lo[j] - params.yz_coef*params.pz_hi[k]
                dyzHL = - params.yz_coef*params.py_hi[j] - params.yz_coef*params.pz_lo[k]
                dyzHH = - params.yz_coef*params.py_hi[j] - params.yz_coef*params.pz_hi[k]

                eyzL = params.w3a*params.py[j] - params.xz_coef*params.pz_lo[k]
                eyzH = params.w3a*params.py[j] - params.xz_coef*params.pz_hi[k]
                fyzL = - params.xy_coef*params.py_lo[j] + params.w3a*params.pz[k]
                fyzH = - params.xy_coef*params.py_hi[j] + params.w3a*params.pz[k]
                gyzLL = params.py_lo[j] + params.pz_lo[k]
                gyzLH = params.py_lo[j] + params.pz_hi[k]
                gyzHL = params.py_hi[j] + params.pz_lo[k]
                gyzHH = params.py_hi[j] + params.pz_hi[k]

                for i = 1:nx
                    # Load wn_window + x_window
                    for kk=1:3
                        load_z = k+kk-2 > 0 && k+kk-2<=nz;
                        for jj=1:3
                            load_y = j+jj-2 > 0 && j+jj-2<=ny;
                            for ii=1:3
                                load_x = i+ii-2 > 0 && i+ii-2<=nx;
                                if load_x & load_y & load_z
                                    @inbounds x_window[ii,jj,kk] = x[i+ii-2,j+jj-2,k+kk-2]
                                    @inbounds wn_window[ii,jj,kk] = wn[i+ii-2,j+jj-2,k+kk-2]
                                else
                                    @inbounds x_window[ii,jj,kk] = zero_x
                                    @inbounds wn_window[ii,jj,kk] = zero_w
                                end
                            end
                        end
                    end

                    coef[M,N,N] = is_deriv_mode*(params.cx*params.px_lo[i] + c1) - params.wm2 * wn_window[M,N,N]
                    coef[P,N,N] = is_deriv_mode*(params.cx*params.px_hi[i] + c1) - params.wm2 * wn_window[P,N,N]
                    coef[N,M,N] = is_deriv_mode*(params.xy_coef*params.px[i] + c2_lo) - params.wm2 * wn_window[N,M,N]
                    coef[N,P,N] = is_deriv_mode*(params.xy_coef*params.px[i] + c2_hi) - params.wm2 * wn_window[N,P,N]
                    coef[N,N,M] = is_deriv_mode*(params.xz_coef*params.px[i] + c3_lo) - params.wm2 * wn_window[N,N,M]
                    coef[N,N,P] = is_deriv_mode*(params.xz_coef*params.px[i] + c3_hi) - params.wm2 * wn_window[N,N,P]

                    coef[N,M,M] = is_deriv_mode*(params.w3a*params.px[i] + dyzLL) - params.wm3 * wn_window[N,M,M]
                    coef[N,M,P] = is_deriv_mode*(params.w3a*params.px[i] + dyzLH) - params.wm3 * wn_window[N,M,P]
                    coef[N,P,M] = is_deriv_mode*(params.w3a*params.px[i] + dyzHL) - params.wm3 * wn_window[N,P,M]
                    coef[N,P,P] = is_deriv_mode*(params.w3a*params.px[i] + dyzHH) - params.wm3 * wn_window[N,P,P]
                    coef[M,N,M] = is_deriv_mode*(-params.xz_coef*params.px_lo[i] + eyzL) - params.wm3 * wn_window[M,N,M]
                    coef[M,N,P] = is_deriv_mode*(-params.xz_coef*params.px_lo[i] + eyzH) - params.wm3 * wn_window[M,N,P]
                    coef[P,N,M] = is_deriv_mode*(-params.xz_coef*params.px_hi[i] + eyzL) - params.wm3 * wn_window[P,N,M]
                    coef[P,N,P] = is_deriv_mode*(-params.xz_coef*params.px_hi[i] + eyzH) - params.wm3 * wn_window[P,N,P]
                    coef[M,M,N] = is_deriv_mode*(-params.xy_coef*params.px_lo[i] + fyzL) - params.wm3 * wn_window[M,M,N]
                    coef[M,P,N] = is_deriv_mode*(-params.xy_coef*params.px_lo[i] + fyzH) - params.wm3 * wn_window[M,P,N]
                    coef[P,M,N] = is_deriv_mode*(-params.xy_coef*params.px_hi[i] + fyzL) - params.wm3 * wn_window[P,M,N]
                    coef[P,P,N] = is_deriv_mode*(-params.xy_coef*params.px_hi[i] + fyzH) - params.wm3 * wn_window[P,P,N]
                    coef[M,M,M] = is_deriv_mode*(-params.w3a*(params.px_lo[i] + gyzLL)) - params.wm4 * wn_window[M,M,M]
                    coef[M,M,P] = is_deriv_mode*(-params.w3a*(params.px_lo[i] + gyzLH)) - params.wm4 * wn_window[M,M,P]
                    coef[M,P,M] = is_deriv_mode*(-params.w3a*(params.px_lo[i] + gyzHL)) - params.wm4 * wn_window[M,P,M]
                    coef[M,P,P] = is_deriv_mode*(-params.w3a*(params.px_lo[i] + gyzHH)) - params.wm4 * wn_window[M,P,P]
                    coef[P,M,M] = is_deriv_mode*(-params.w3a*(params.px_hi[i] + gyzLL)) - params.wm4 * wn_window[P,M,M]
                    coef[P,M,P] = is_deriv_mode*(-params.w3a*(params.px_hi[i] + gyzLH)) - params.wm4 * wn_window[P,M,P]
                    coef[P,P,M] = is_deriv_mode*(-params.w3a*(params.px_hi[i] + gyzHL)) - params.wm4 * wn_window[P,P,M]
                    coef[P,P,P] = is_deriv_mode*(-params.w3a*(params.px_hi[i] + gyzHH)) - params.wm4 * wn_window[P,P,P]
                    coef[N,N,N] = is_deriv_mode*(-params.cx*params.px[i] - params.cy*params.py[j] - params.cz*params.pz[k]) + params.cNNN*wn_window[N,N,N]
                    t = complex(0.0,0.0);
                    for kk=1:3
                        for jj=1:3
                            for ii=1:3
                                @inbounds t += coef[ii,jj,kk]*x_window[ii,jj,kk];
                            end
                        end
                    end
                    y[i,j,k] = t;
                end
            end
        end
    end
end

function helm3d_operto_mvp_adj_impl!(wn, x, y, n, params, deriv_mode)
    M, N, P = 1, 2, 3
    zero_x = complex(0.0,0.0)
    zero_w = zero(eltype(wn))
    nx,ny,nz = n
    is_deriv_mode = deriv_mode ? complex(0.0,0.0) : complex(1.0,0.0)
    num_blocks = Threads.nthreads()
    block_size = ceil(Int64,nz/num_blocks)
    Threads.@threads for i=1:num_blocks
        z_idx = (i-1)*block_size+1:min(nz,i*block_size)
        coef = zeros(Complex{Float64},3,3,3)
        x_window = zeros(Complex{Float64},3,3,3)

        # (i+1,j+1,k+1) is the current index for the current point
        # with respect to the pml functions
        for k = z_idx
            kM,kN,kP = k,k+1,k+2
            for j=1:ny
                jM,jN,jP = j,j+1,j+2
                for i=1:nx
                    wNNN = wn[i,j,k]
                    for kk=1:3
                        load_z = k+kk-2 > 0 && k+kk-2<=nz;
                        for jj=1:3
                            load_y = j+jj-2 > 0 && j+jj-2<=ny;
                            for ii=1:3
                                load_x = i+ii-2 > 0 && i+ii-2<=nx;
                                if load_x & load_y & load_z
                                @inbounds x_window[ii,jj,kk] = x[i+ii-2,j+jj-2,k+kk-2]
                                else
                                 @inbounds x_window[ii,jj,kk] = zero_x
                                end
                            end
                        end
                    end
                    iM,iN,iP = i,i+1,i+2
                    coef[M,M,M] = -params.wm4*wNNN + is_deriv_mode*(-params.w3a*(params.px_hi[iM] + params.py_hi[jM] + params.pz_hi[kM]))
                    coef[N,M,M] = -params.wm3*wNNN + is_deriv_mode*((-params.yz_coef) * (params.pz_hi[kM] + params.py_hi[jM]) + params.w3a*params.px[iN])
                    coef[P,M,M] = -params.wm4*wNNN + is_deriv_mode*(-params.w3a*( params.px_lo[iP] + params.py_hi[jM] + params.pz_hi[kM]))
                    coef[M,N,M] = -params.wm3*wNNN + is_deriv_mode*(-params.xz_coef * (params.px_hi[iM] + params.pz_hi[kM]) + params.w3a*params.py[jN])
                    coef[N,N,M] = -params.wm2*wNNN + is_deriv_mode*(params.cz*params.pz_hi[kM] + params.yz_coef*params.py[jN] + params.xz_coef*params.px[iN])
                    coef[P,N,M] = -params.wm3*wNNN + is_deriv_mode*(-params.xz_coef * (params.pz_hi[kM] + params.px_lo[iP]) + params.w3a*params.py[jN])
                    coef[M,P,M] = -params.wm4*wNNN + is_deriv_mode*(-params.w3a*( params.px_hi[iM] + params.py_lo[jP] + params.pz_hi[kM] ));
                    coef[N,P,M] = -params.wm3*wNNN + is_deriv_mode*(-params.yz_coef * (params.py_lo[jP] + params.pz_hi[kM]) + params.w3a*params.px[iN])
                    coef[P,P,M] = -params.wm4*wNNN + is_deriv_mode*(-params.w3a*( params.px_lo[iP] + params.py_lo[jP] + params.pz_hi[kM]))
                    coef[M,M,N] = -params.wm3*wNNN + is_deriv_mode*(-params.xy_coef * (params.px_hi[iM] + params.py_hi[jM]) + params.w3a*params.pz[kN])
                    coef[N,M,N] = -params.wm2*wNNN + is_deriv_mode*(params.cy*params.py_hi[jM] + params.yz_coef*params.pz[kN] + params.xy_coef*params.px[iN])
                    coef[P,M,N] = -params.wm3*wNNN + is_deriv_mode*(-params.xy_coef * (params.px_lo[iP] + params.py_hi[jM]) + params.w3a*params.pz[kN]	)
                    coef[M,N,N] = -params.wm2*wNNN + is_deriv_mode*(params.cx*params.px_hi[iM] + params.xz_coef*params.pz[kN] + params.xy_coef*params.py[jN])
                    coef[N,N,N] = params.cNNN*wNNN - is_deriv_mode*( params.cx*params.px[iN] + params.cy*params.py[jN] + params.cz*params.pz[kN])
                    coef[P,N,N] = -params.wm2*wNNN + is_deriv_mode*(params.cx*params.px_lo[iP] + params.xz_coef*params.pz[kN] + params.xy_coef*params.py[jN])
                    coef[M,P,N] = -params.wm3*wNNN + is_deriv_mode*(-params.xy_coef * (params.px_hi[iM] + params.py_lo[jP]) + params.w3a*params.pz[kN])
                    coef[N,P,N] = -params.wm2*wNNN + is_deriv_mode*(params.cy*params.py_lo[jP] + params.yz_coef*params.pz[kN] + params.xy_coef*params.px[iN])
                    coef[P,P,N] = -params.wm3*wNNN + is_deriv_mode*(-params.xy_coef * (params.px_lo[iP] + params.py_lo[jP]) + params.w3a*params.pz[kN])
                    coef[M,M,P] = -params.wm4*wNNN + is_deriv_mode*(-params.w3a*( params.px_hi[iM] + params.py_hi[jM] + params.pz_lo[kP] ))
                    coef[N,M,P] = -params.wm3*wNNN + is_deriv_mode*(-params.yz_coef * (params.py_hi[jM] + params.pz_lo[kP]) + params.w3a*params.px[iN])
                    coef[P,M,P] = -params.wm4*wNNN + is_deriv_mode*(-params.w3a*( params.px_lo[iP] + params.py_hi[jM] + params.pz_lo[kP]))
                    coef[M,N,P] = -params.wm3*wNNN + is_deriv_mode*(-params.xz_coef * (params.pz_lo[kP] + params.px_hi[iM]) + params.w3a*params.py[jN])
                    coef[N,N,P] = -params.wm2*wNNN + is_deriv_mode*(params.cz*params.pz_lo[kP] + params.yz_coef*params.py[jN] + params.xz_coef*params.px[iN])
                    coef[P,N,P] = -params.wm3*wNNN + is_deriv_mode*(-params.xz_coef * (params.pz_lo[kP] + params.px_lo[iP]) + params.w3a*params.py[jN])
                    coef[M,P,P] = -params.wm4*wNNN + is_deriv_mode*(-params.w3a*( params.px_hi[iM] + params.py_lo[jP] + params.pz_lo[kP]))
                    coef[N,P,P] = -params.wm3*wNNN + is_deriv_mode*(-params.yz_coef * (params.pz_lo[kP] + params.py_lo[jP]) + params.w3a*params.px[iN])
                    coef[P,P,P] = -params.wm4*wNNN + is_deriv_mode*(-params.w3a*( params.px_lo[iP] + params.py_lo[jP] + params.pz_lo[kP]))

                    t = complex(0.0,0.0);
                    for kk=1:3
                        for jj=1:3
                            for ii=1:3
                                @inbounds t += conj(coef[ii,jj,kk])*x_window[ii,jj,kk];
                            end
                        end
                    end
                    y[i,j,k] = t;
                end
            end
        end
    end
end


function helm3d_operto_mvp_mt(wn,Δ,n,freq,npml,x; forw_mode::Bool=true, deriv_mode::Bool=false)
    M,N,P = 1,2,3

    nx,ny,nz = n[1],n[2],n[3]
    hx,hy,hz = Δ[1]^2,Δ[2]^2,Δ[3]^2
    hyz  = hy + hz
    hxy = hy + hx
    hxz = hx + hz
    hxyz = hx + hy + hz

    npxlo,npxhi,npylo,npyhi,npzlo,npzhi = npml[1,1],npml[2,1],npml[1,2],npml[2,2],npml[1,3],npml[2,3]

    y = zeros(Complex{Float64},nx,ny,nz)

    # Create PML functions
    px_lo,px_hi = pml_func(nx,npxlo,npxhi)
    if ~forw_mode
        prepend!(px_lo,[px_lo[1]])
        append!(px_lo,[px_lo[end]])
        prepend!(px_hi,[px_hi[1]])
        append!(px_hi,[px_hi[end]])
    end
    px = px_lo+px_hi
    py_lo,py_hi = pml_func(ny,npylo,npyhi)
    if ~forw_mode
        prepend!(py_lo,[py_lo[1]])
        append!(py_lo,[py_lo[end]])
        prepend!(py_hi,[py_hi[1]])
        append!(py_hi,[py_hi[end]])
    end

    py = py_lo+py_hi
    pz_lo,pz_hi = pml_func(nz,npzlo,npzhi)
    if ~forw_mode
        prepend!(pz_lo,[pz_lo[1]])
        append!(pz_lo,[pz_lo[end]])
        prepend!(pz_hi,[pz_hi[1]])
        append!(pz_hi,[pz_hi[end]])
    end

    pz = pz_lo+pz_hi

    # Weights
    const w1  = 1.8395262e-5
    const w2  = 0.29669233333333334
    const w3  = 0.02747615
    const wm1 = 0.49649658
    const wm2 = 0.07516874999999999
    const wm3 = 0.004373916666666667
    const wm4 = 5.690375e-7
    w3a = 2*(w3*3/(4*hxyz))

    cx = - (w1/hx + w2/hx + w2/hxz + w2/hxy + 4*w3a)
    cy = - (w1/hy + w2/hy + w2/hyz + w2/hxy + 4*w3a)
    cz = - (w1/hz + w2/hz + w2/hxz + w2/hyz + 4*w3a)
    cNNN = - (w1 + 3*w2 + (8*w3a*hxyz)/3 + wm1-1)

    xz_coef = w2/(2*hxz)
    xy_coef = w2/(2*hxy)
    yz_coef = w2/(2*hyz)
    params = helm3d_params{Float64}(w3a,wm2,wm3,wm4,cNNN,cx,cy,cz,xy_coef,xz_coef,yz_coef,px_lo,px_hi,px,py_lo,py_hi,py,pz_lo,pz_hi,pz)
    if forw_mode
        helm3d_operto_mvp_forw_impl!(wn, x, y, n, params, deriv_mode)
    else
        helm3d_operto_mvp_adj_impl!(wn, x, y, n, params, deriv_mode)
    end
    return vec(y)
end


function helm3d_operto_mvp(wn::Union{AbstractArray{F,3},AbstractArray{Complex{F},3}},Δ::AbstractArray{F,1},n::AbstractArray{I,1},freq::Union{F,Complex{F}},npml::AbstractArray{I,2},x::AbstractArray{Complex{F},3};forw_mode::Bool=true,deriv_mode::Bool=false) where {F<:Real,I<:Integer}


    M,N,P = 1,2,3

    nx,ny,nz = n[1],n[2],n[3]
    hx,hy,hz = Δ[1]^2,Δ[2]^2,Δ[3]^2
    hyz  = hy + hz
    hxy = hy + hx
    hxz = hx + hz
    hxyz = hx + hy + hz


    npxlo,npxhi,npylo,npyhi,npzlo,npzhi = npml[1,1],npml[2,1],npml[1,2],npml[2,2],npml[1,3],npml[2,3]

    y = zeros(Complex{F},tuple(n...))

    # Create PML functions
    px_lo,px_hi = pml_func(nx,npxlo,npxhi)
    if ~forw_mode
        prepend!(px_lo,[px_lo[1]])
        append!(px_lo,[px_lo[end]])
        prepend!(px_hi,[px_hi[1]])
        append!(px_hi,[px_hi[end]])
    end
    px = px_lo+px_hi
    py_lo,py_hi = pml_func(ny,npylo,npyhi)
    if ~forw_mode
        prepend!(py_lo,[py_lo[1]])
        append!(py_lo,[py_lo[end]])
        prepend!(py_hi,[py_hi[1]])
        append!(py_hi,[py_hi[end]])
    end

    py = py_lo+py_hi
    pz_lo,pz_hi = pml_func(nz,npzlo,npzhi)
    if ~forw_mode
        prepend!(pz_lo,[pz_lo[1]])
        append!(pz_lo,[pz_lo[end]])
        prepend!(pz_hi,[pz_hi[1]])
        append!(pz_hi,[pz_hi[end]])
    end

    pz = pz_lo+pz_hi

    # Weights
    const w1  = 1.8395262e-5
    const w2  = 0.29669233333333334
    const w3  = 0.02747615
    const wm1 = 0.49649658
    const wm2 = 0.07516874999999999
    const wm3 = 0.004373916666666667
    const wm4 = 5.690375e-7
    w3a = 2*(w3*3/(4*hxyz))::F

    cx = - (w1/hx + w2/hx + w2/hxz + w2/hxy + 4*w3a)
    cy = - (w1/hy + w2/hy + w2/hyz + w2/hxy + 4*w3a)
    cz = - (w1/hz + w2/hz + w2/hxz + w2/hyz + 4*w3a)
    cNNN = - (w1 + 3*w2 + (8*w3a*hxyz)/3 + wm1-1)

    xz_coef = w2/(2*hxz)
    xy_coef = w2/(2*hxy)
    yz_coef = w2/(2*hyz)
    wn_window = zeros(F,3,3,3)
    coef = zeros(Complex{F},3,3,3)
    x_window = zeros(eltype(x),3,3,3)
    zero_x = complex(0.0,0.0)
    zero_w = zero(eltype(wn))
    is_deriv_mode = deriv_mode ? complex(0.0,0.0) : complex(1.0,0.0)
    if forw_mode
        for k = 1:nz
            cxz = xz_coef*pz[k]
	        cxzlo = xz_coef*pz_lo[k]
	        cxzhi = xz_coef*pz_hi[k]
	        cyz = yz_coef*pz[k]
	        cyzlo = yz_coef*pz_lo[k]
	        cyzhi = yz_coef*pz_hi[k]
            for j = 1:ny
                c1 = xy_coef*py[j] + cxz
		        c2_lo = cy*py_lo[j] + cyz
		        c2_hi = cy*py_hi[j] + cyz
		        c3_lo = yz_coef*py[j] + cz*pz_lo[k]
		        c3_hi = yz_coef*py[j] + cz*pz_hi[k]

                dyzLL = - yz_coef*py_lo[j] - yz_coef*pz_lo[k]
                dyzLH = - yz_coef*py_lo[j] - yz_coef*pz_hi[k]
                dyzHL = - yz_coef*py_hi[j] - yz_coef*pz_lo[k]
                dyzHH = - yz_coef*py_hi[j] - yz_coef*pz_hi[k]

                eyzL = w3a*py[j] - xz_coef*pz_lo[k]
                eyzH = w3a*py[j] - xz_coef*pz_hi[k]
                fyzL = - xy_coef*py_lo[j] + w3a*pz[k]
                fyzH = - xy_coef*py_hi[j] + w3a*pz[k]
                gyzLL = py_lo[j] + pz_lo[k]
                gyzLH = py_lo[j] + pz_hi[k]
                gyzHL = py_hi[j] + pz_lo[k]
                gyzHH = py_hi[j] + pz_hi[k]

                for i = 1:nx
                    # Load wn_window + x_window
                    for kk=1:3
                        load_z = k+kk-2 > 0 && k+kk-2<=nz;
                        for jj=1:3
                            load_y = j+jj-2 > 0 && j+jj-2<=ny;
                            for ii=1:3
                                load_x = i+ii-2 > 0 && i+ii-2<=nx;
                                if load_x & load_y & load_z
                                    @inbounds x_window[ii,jj,kk] = x[i+ii-2,j+jj-2,k+kk-2]
                                    @inbounds wn_window[ii,jj,kk] = wn[i+ii-2,j+jj-2,k+kk-2]
                                else
                                    @inbounds x_window[ii,jj,kk] = zero_x
                                    @inbounds wn_window[ii,jj,kk] = zero_w
                                end
                            end
                        end
                    end

			        coef[M,N,N] = is_deriv_mode*(cx*px_lo[i] + c1) - wm2 * wn_window[M,N,N]
			        coef[P,N,N] = is_deriv_mode*(cx*px_hi[i] + c1) - wm2 * wn_window[P,N,N]
			        coef[N,M,N] = is_deriv_mode*(xy_coef*px[i] + c2_lo) - wm2 * wn_window[N,M,N]
			        coef[N,P,N] = is_deriv_mode*(xy_coef*px[i] + c2_hi) - wm2 * wn_window[N,P,N]
			        coef[N,N,M] = is_deriv_mode*(xz_coef*px[i] + c3_lo) - wm2 * wn_window[N,N,M]
			        coef[N,N,P] = is_deriv_mode*(xz_coef*px[i] + c3_hi) - wm2 * wn_window[N,N,P]

			        coef[N,M,M] = is_deriv_mode*(w3a*px[i] + dyzLL) - wm3 * wn_window[N,M,M]
			        coef[N,M,P] = is_deriv_mode*(w3a*px[i] + dyzLH) - wm3 * wn_window[N,M,P]
			        coef[N,P,M] = is_deriv_mode*(w3a*px[i] + dyzHL) - wm3 * wn_window[N,P,M]
			        coef[N,P,P] = is_deriv_mode*(w3a*px[i] + dyzHH) - wm3 * wn_window[N,P,P]
			        coef[M,N,M] = is_deriv_mode*(-xz_coef*px_lo[i] + eyzL) - wm3 * wn_window[M,N,M]
			        coef[M,N,P] = is_deriv_mode*(-xz_coef*px_lo[i] + eyzH) - wm3 * wn_window[M,N,P]
			        coef[P,N,M] = is_deriv_mode*(-xz_coef*px_hi[i] + eyzL) - wm3 * wn_window[P,N,M]
			        coef[P,N,P] = is_deriv_mode*(-xz_coef*px_hi[i] + eyzH) - wm3 * wn_window[P,N,P]
			        coef[M,M,N] = is_deriv_mode*(-xy_coef*px_lo[i] + fyzL) - wm3 * wn_window[M,M,N]
			        coef[M,P,N] = is_deriv_mode*(-xy_coef*px_lo[i] + fyzH) - wm3 * wn_window[M,P,N]
			        coef[P,M,N] = is_deriv_mode*(-xy_coef*px_hi[i] + fyzL) - wm3 * wn_window[P,M,N]
			        coef[P,P,N] = is_deriv_mode*(-xy_coef*px_hi[i] + fyzH) - wm3 * wn_window[P,P,N]
			        coef[M,M,M] = is_deriv_mode*(-w3a*(px_lo[i] + gyzLL)) - wm4 * wn_window[M,M,M]
			        coef[M,M,P] = is_deriv_mode*(-w3a*(px_lo[i] + gyzLH)) - wm4 * wn_window[M,M,P]
			        coef[M,P,M] = is_deriv_mode*(-w3a*(px_lo[i] + gyzHL)) - wm4 * wn_window[M,P,M]
			        coef[M,P,P] = is_deriv_mode*(-w3a*(px_lo[i] + gyzHH)) - wm4 * wn_window[M,P,P]
			        coef[P,M,M] = is_deriv_mode*(-w3a*(px_hi[i] + gyzLL)) - wm4 * wn_window[P,M,M]
			        coef[P,M,P] = is_deriv_mode*(-w3a*(px_hi[i] + gyzLH)) - wm4 * wn_window[P,M,P]
			        coef[P,P,M] = is_deriv_mode*(-w3a*(px_hi[i] + gyzHL)) - wm4 * wn_window[P,P,M]
			        coef[P,P,P] = is_deriv_mode*(-w3a*(px_hi[i] + gyzHH)) - wm4 * wn_window[P,P,P]
                    coef[N,N,N] = is_deriv_mode*(-cx*px[i] - cy*py[j] - cz*pz[k]) + cNNN*wn_window[N,N,N]
                    t = complex(0.0,0.0);
                    for kk=1:3
                        for jj=1:3
                            for ii=1:3
                                @inbounds t += coef[ii,jj,kk]*x_window[ii,jj,kk];
                            end
                        end
                    end
                    y[i,j,k] = t;
                end
            end
        end
    else
    # (i+1,j+1,k+1) is the current index for the current point
    # with respect to the pml functions
        for k=1:nz
            kM,kN,kP = k,k+1,k+2
            for j=1:ny
                jM,jN,jP = j,j+1,j+2
                for i=1:nx
                    wNNN = wn[i,j,k]
                    for kk=1:3
                        load_z = k+kk-2 > 0 && k+kk-2<=nz;
                        for jj=1:3
                            load_y = j+jj-2 > 0 && j+jj-2<=ny;
                            for ii=1:3
                                load_x = i+ii-2 > 0 && i+ii-2<=nx;
                                if load_x & load_y & load_z
                                @inbounds x_window[ii,jj,kk] = x[i+ii-2,j+jj-2,k+kk-2]
                                else
                                 @inbounds x_window[ii,jj,kk] = zero_x
                                end
                            end
                        end
                    end
                    iM,iN,iP = i,i+1,i+2
                    coef[M,M,M] = -wm4*wNNN + is_deriv_mode*(-w3a*(px_hi[iM] + py_hi[jM] + pz_hi[kM]))
                    coef[N,M,M] = -wm3*wNNN + is_deriv_mode*((-yz_coef) * (pz_hi[kM] + py_hi[jM]) + w3a*px[iN])
                    coef[P,M,M] = -wm4*wNNN + is_deriv_mode*(-w3a*( px_lo[iP] + py_hi[jM] + pz_hi[kM]))
                    coef[M,N,M] = -wm3*wNNN + is_deriv_mode*(-xz_coef * (px_hi[iM] + pz_hi[kM]) + w3a*py[jN])

                    coef[N,N,M] = -wm2*wNNN + is_deriv_mode*(cz*pz_hi[kM] + yz_coef*py[jN] + xz_coef*px[iN])
                    coef[P,N,M] = -wm3*wNNN + is_deriv_mode*(-xz_coef * (pz_hi[kM] + px_lo[iP]) + w3a*py[jN])
                    coef[M,P,M] = -wm4*wNNN + is_deriv_mode*(-w3a*( px_hi[iM] + py_lo[jP] + pz_hi[kM] ));
                    coef[N,P,M] = -wm3*wNNN + is_deriv_mode*(-yz_coef * (py_lo[jP] + pz_hi[kM]) + w3a*px[iN])
                    coef[P,P,M] = -wm4*wNNN + is_deriv_mode*(-w3a*( px_lo[iP] + py_lo[jP] + pz_hi[kM]))
                    coef[M,M,N] = -wm3*wNNN + is_deriv_mode*(-xy_coef * (px_hi[iM] + py_hi[jM]) + w3a*pz[kN])
                    coef[N,M,N] = -wm2*wNNN + is_deriv_mode*(cy*py_hi[jM] + yz_coef*pz[kN] + xy_coef*px[iN])
                    coef[P,M,N] = -wm3*wNNN + is_deriv_mode*(-xy_coef * (px_lo[iP] + py_hi[jM]) + w3a*pz[kN]	)
                    coef[M,N,N] = -wm2*wNNN + is_deriv_mode*(cx*px_hi[iM] + xz_coef*pz[kN] + xy_coef*py[jN])
                    coef[N,N,N] = cNNN*wNNN - is_deriv_mode*( cx*px[iN] + cy*py[jN] + cz*pz[kN])
                    coef[P,N,N] = -wm2*wNNN + is_deriv_mode*(cx*px_lo[iP] + xz_coef*pz[kN] + xy_coef*py[jN])

                    coef[M,P,N] = -wm3*wNNN + is_deriv_mode*(-xy_coef * (px_hi[iM] + py_lo[jP]) + w3a*pz[kN])
                    coef[N,P,N] = -wm2*wNNN + is_deriv_mode*(cy*py_lo[jP] + yz_coef*pz[kN] + xy_coef*px[iN])
                    coef[P,P,N] = -wm3*wNNN + is_deriv_mode*(-xy_coef * (px_lo[iP] + py_lo[jP]) + w3a*pz[kN])
                    coef[M,M,P] = -wm4*wNNN + is_deriv_mode*(-w3a*( px_hi[iM] + py_hi[jM] + pz_lo[kP] ))
                    coef[N,M,P] = -wm3*wNNN + is_deriv_mode*(-yz_coef * (py_hi[jM] + pz_lo[kP]) + w3a*px[iN])
                    coef[P,M,P] = -wm4*wNNN + is_deriv_mode*(-w3a*( px_lo[iP] + py_hi[jM] + pz_lo[kP]))
                    coef[M,N,P] = -wm3*wNNN + is_deriv_mode*(-xz_coef * (pz_lo[kP] + px_hi[iM]) + w3a*py[jN])
                    coef[N,N,P] = -wm2*wNNN + is_deriv_mode*(cz*pz_lo[kP] + yz_coef*py[jN] + xz_coef*px[iN])
                    coef[P,N,P] = -wm3*wNNN + is_deriv_mode*(-xz_coef * (pz_lo[kP] + px_lo[iP]) + w3a*py[jN])
                    coef[M,P,P] = -wm4*wNNN + is_deriv_mode*(-w3a*( px_hi[iM] + py_lo[jP] + pz_lo[kP]))
                    coef[N,P,P] = -wm3*wNNN + is_deriv_mode*(-yz_coef * (pz_lo[kP] + py_lo[jP]) + w3a*px[iN])
                    coef[P,P,P] = -wm4*wNNN + is_deriv_mode*(-w3a*( px_lo[iP] + py_lo[jP] + pz_lo[kP]))

                    t = complex(0.0,0.0);
                    for kk=1:3
                        for jj=1:3
                            for ii=1:3
                                @inbounds t += conj(coef[ii,jj,kk])*x_window[ii,jj,kk];
                            end
                        end
                    end
                    y[i,j,k] = t;
                end
            end
        end
    end
    return vec(y)
end


function helm3d_operto_matrix(wn::Union{AbstractArray{F,3},AbstractArray{Complex{F},3}},Δ::AbstractArray{F,1},n::AbstractArray{I,1},freq::Union{F,Complex{F}},npml::AbstractArray{I,2}) where {F<:Real,I<:Integer}
    """
    helm3d_operto_matrix(wn,h,nt,npml)

    Curt Da Silva, 2016
    """
    M,N,P = 1,2,3

    nx,ny,nz = n[1],n[2],n[3]
    hx,hy,hz = Δ[1]^2,Δ[2]^2,Δ[3]^2
    hyz  = hy + hz
    hxy = hy + hx
    hxz = hx + hz
    hxyz = hx + hy + hz


    npxlo,npxhi,npylo,npyhi,npzlo,npzhi = npml[1,1],npml[2,1],npml[1,2],npml[2,2],npml[1,3],npml[2,3]

    y = zeros(Complex{F},tuple(n...))

    # Create PML functions
    px_lo,px_hi = pml_func(nx,npxlo,npxhi)
    px = px_lo+px_hi
    py_lo,py_hi = pml_func(ny,npylo,npyhi)
    py = py_lo+py_hi
    pz_lo,pz_hi = pml_func(nz,npzlo,npzhi)
    pz = pz_lo+pz_hi

    # Weights
    w1  = 1.8395262e-5
    w2  = 0.29669233333333334
    w3  = 0.02747615
    wm1 = 0.49649658
    wm2 = 0.07516874999999999
    wm3 = 0.004373916666666667
    wm4 = 5.690375e-7
    w3a = (0.041214225/hxyz)::F

    cx = - (w1/hx + w2/hx + w2/hxz + w2/hxy + 8*w3a)
    cy = - (w1/hy + w2/hy + w2/hyz + w2/hxy + 8*w3a)
    cz = - (w1/hz + w2/hz + w2/hxz + w2/hyz + 8*w3a)
    cNNN = - (w1 + 3*w2 + (16*w3a*hxyz)/3 + wm1-1)

    xz_coef = w2/(2*hxz)
    xy_coef = w2/(2*hxy)
    yz_coef = w2/(2*hyz)
    wn_window = zeros(F,3,3,3)
    coef = zeros(Complex{F},3,3,3,nx,ny,nz)

    for k = 1:nz
        ks = (k > 1)? -1 : 0
        ke = (k < nz)? 1 : 0
        zoff = ks:ke
        for j = 1:ny
            js = (j > 1)? -1 : 0
            je = (j < ny)? 1 : 0
            yoff = js:je
            for i = 1:nx
                is = (i > 1)? -1 : 0
                ie = (i < nx) ? 1 : 0
                xoff = is:ie
                # Load wn_window
			    wn_window[2+xoff,2+yoff,2+zoff] = wn[i+xoff,j+yoff,k+zoff]
                cxz = xz_coef*pz[k]
	            cxzlo = xz_coef*pz_lo[k]
	            cxzhi = xz_coef*pz_hi[k]
	            cyz = yz_coef*pz[k]
	            cyzlo = yz_coef*pz_lo[k]
	            cyzhi = yz_coef*pz_hi[k]

                c1 = xy_coef*py[j] + cxz
		        c2_lo = cy*py_lo[j] + cyz
		        c2_hi = cy*py_hi[j] + cyz
		        c3_lo = yz_coef*py[j] + cz*pz_lo[k]
		        c3_hi = yz_coef*py[j] + cz*pz_hi[k]

			    coef[M,N,N,i,j,k] = cx*px_lo[i] + c1 - wm2 * wn_window[M,N,N]
			    coef[P,N,N,i,j,k] = cx*px_hi[i] + c1 - wm2 * wn_window[P,N,N]
			    coef[N,M,N,i,j,k] = xy_coef*px[i] + c2_lo - wm2 * wn_window[N,M,N]
			    coef[N,P,N,i,j,k] = xy_coef*px[i] + c2_hi - wm2 * wn_window[N,P,N]
			    coef[N,N,M,i,j,k] = xz_coef*px[i] + c3_lo - wm2 * wn_window[N,N,M]
			    coef[N,N,P,i,j,k] = xz_coef*px[i] + c3_hi - wm2 * wn_window[N,N,P]

			    coef[N,M,M,i,j,k] = 2*w3a*px[i] - yz_coef*py_lo[j] - yz_coef*pz_lo[k] - wm3 * wn_window[N,M,M]
			    coef[N,M,P,i,j,k] = 2*w3a*px[i] - yz_coef*py_lo[j] - yz_coef*pz_hi[k] - wm3 * wn_window[N,M,P]
			    coef[N,P,M,i,j,k] = 2*w3a*px[i] - yz_coef*py_hi[j] - yz_coef*pz_lo[k] - wm3 * wn_window[N,P,M]
			    coef[N,P,P,i,j,k] = 2*w3a*px[i] - yz_coef*py_hi[j] - yz_coef*pz_hi[k] - wm3 * wn_window[N,P,P]
			    coef[M,N,M,i,j,k] = -xz_coef*px_lo[i] + 2*w3a*py[j] - xz_coef*pz_lo[k] - wm3 * wn_window[M,N,M]
			    coef[M,N,P,i,j,k] = -xz_coef*px_lo[i] + 2*w3a*py[j] - xz_coef*pz_hi[k] - wm3 * wn_window[M,N,P]
			    coef[P,N,M,i,j,k] = -xz_coef*px_hi[i] + 2*w3a*py[j] - xz_coef*pz_lo[k] - wm3 * wn_window[P,N,M]
			    coef[P,N,P,i,j,k] = -xz_coef*px_hi[i] + 2*w3a*py[j] - xz_coef*pz_hi[k] - wm3 * wn_window[P,N,P]
			    coef[M,M,N,i,j,k] = -xy_coef*px_lo[i] - xy_coef*py_lo[j] + 2*w3a*pz[k] - wm3 * wn_window[M,M,N]
			    coef[M,P,N,i,j,k] = -xy_coef*px_lo[i] - xy_coef*py_hi[j] + 2*w3a*pz[k] - wm3 * wn_window[M,P,N]
			    coef[P,M,N,i,j,k] = -xy_coef*px_hi[i] - xy_coef*py_lo[j] + 2*w3a*pz[k] - wm3 * wn_window[P,M,N]
			    coef[P,P,N,i,j,k] = -xy_coef*px_hi[i] - xy_coef*py_hi[j] + 2*w3a*pz[k] - wm3 * wn_window[P,P,N]
			    coef[M,M,M,i,j,k] = -2*w3a*(px_lo[i] + py_lo[j] + pz_lo[k]) - wm4 * wn_window[M,M,M]
			    coef[M,M,P,i,j,k] = -2*w3a*(px_lo[i] + py_lo[j] + pz_hi[k]) - wm4 * wn_window[M,M,P]
			    coef[M,P,M,i,j,k] = -2*w3a*(px_lo[i] + py_hi[j] + pz_lo[k]) - wm4 * wn_window[M,P,M]
			    coef[M,P,P,i,j,k] = -2*w3a*(px_lo[i] + py_hi[j] + pz_hi[k]) - wm4 * wn_window[M,P,P]
			    coef[P,M,M,i,j,k] = -2*w3a*(px_hi[i] + py_lo[j] + pz_lo[k]) - wm4 * wn_window[P,M,M]
			    coef[P,M,P,i,j,k] = -2*w3a*(px_hi[i] + py_lo[j] + pz_hi[k]) - wm4 * wn_window[P,M,P]
			    coef[P,P,M,i,j,k] = -2*w3a*(px_hi[i] + py_hi[j] + pz_lo[k]) - wm4 * wn_window[P,P,M]
			    coef[P,P,P,i,j,k] = -2*w3a*(px_hi[i] + py_hi[j] + pz_hi[k]) - wm4 * wn_window[P,P,P]
			    coef[N,N,N,i,j,k] = -cx*px[i] - cy*py[j] - cz*pz[k] + cNNN*wn_window[N,N,N]

            end
        end
    end
    # Zero out coefficients on the boundary layers
    for i=[M N P]
        for j=[M N P]
            for k=[M N P]
                if i==M
                    coef[i,j,k,1,:,:] = 0
                elseif i==P
                    coef[i,j,k,end,:,:] = 0
                end
                if j==M
                    coef[i,j,k,:,1,:] = 0
                elseif j==P
                    coef[i,j,k,:,end,:] = 0
                end
                if k==M
                    coef[i,j,k,:,:,1] = 0
                elseif k==P
                    coef[i,j,k,:,:,end] = 0
                end
            end
        end
    end


    coef_tup = Array{Array{Complex{F},1},1}()
    offset_tup = Array{I,1}()
    Nel = nx*ny*nz
    t = 1
    for k=[M N P]
        for j=[M N P]
            for i=[M N P]
                idx_start = 1
                idx_end = Nel
                offset = 0
                # For the coefficients before the diagonal, in lexographial organization
                # Remove the coefficient
                if t < 14
                    if i==M
                        idx_start += 1
                        offset -= 1
                    elseif i==P
                        idx_start -= 1
                        offset += 1
                    end
                    if j==M
                        idx_start += nx
                        offset -= nx
                    elseif j==P
                        idx_start -= nx
                        offset += nx
                    end
                    if k==M
                        idx_start += nx*ny
                        offset -= nx*ny
                    end
                elseif t > 14
                    if i==M
                        idx_end += 1
                        offset -= 1
                    elseif i==P
                        idx_end -= 1
                        offset += 1
                    end
                    if j==M
                        idx_end += nx
                        offset -= nx
                    elseif j==P
                        idx_end -= nx
                        offset += nx
                    end
                    if k==P
                        idx_end -= nx*ny
                        offset += nx*ny
                    end
                end
                push!(coef_tup,vec(coef[i,j,k,:,:,:])[idx_start:idx_end])
                push!(offset_tup,offset)

                t += 1
            end
        end
    end

    H = spdiagm(tuple(coef_tup...),tuple(offset_tup...),Nel,Nel)

end


function helm3d_std_7pt(v::Union{AbstractArray{F,3},AbstractArray{Complex{F},3}},Δ::AbstractArray{F,1},n::AbstractArray{I,1},freq::Union{F,Complex{F}},npml::AbstractArray{I,2},x::AbstractArray{Complex{F},3};forw_mode::Bool=true,deriv_mode::Bool=true) where {F<:Real,I<:Integer}
    nx,ny,nz = n[1],n[2],n[3]
    hx,hy,hz = Δ[1],Δ[2],Δ[3]
    hx_isq = 1/hx^2
    hy_isq = 1/hy^2
    hz_isq = 1/hz^2
    M = 1
    N = 2
    P = 3
    ω2 = (2*π*freq)^2
    npxlo,npxhi,npylo,npyhi,npzlo,npzhi = npml[1,1],npml[2,1],npml[1,2],npml[2,2],npml[1,3],npml[2,3]
    maxnpx = max(npxlo,npxhi)
    maxnpy = max(npylo,npyhi)
    maxnpz = max(npzlo,npzhi)
    ξx = x->ξ(x,nx,npxlo,npxhi,maxnpx,freq)
    ξy = y->ξ(y,ny,npylo,npyhi,maxnpy,freq)
    ξz = z->ξ(z,nz,npzlo,npzhi,maxnpz,freq)
    coef = zeros(Complex{F},3,3,3)
    x_window = zeros(Complex{F},3,3,3)
    y = zeros(Complex{F},n...)

    for k=1:nz
        ξzM = ξz(k-0.5)
        ξzN = ξz(k)
        ξzP = ξz(k+0.5)
        ξzlo = ξzM*ξzN*hz_isq
        ξzhi = ξzP*ξzN*hz_isq
        coef[N,N,M] = -ξzlo
        coef[N,N,P] = -ξzhi
        for j=1:ny
            ξyM = ξy(j-0.5)
            ξyN = ξy(j)
            ξyP = ξy(j+0.5)
            ξylo = ξyM*ξyN*hy_isq
            ξyhi = ξyP*ξyN*hy_isq
            coef[N,M,N] = -ξylo
            coef[N,P,N] = -ξyhi
            for i=1:nx
                 for kk=1:3
                    load_z = k+kk-2 > 0 && k+kk-2<=nz;
                    for jj=1:3
                        load_y = j+jj-2 > 0 && j+jj-2<=ny;
                        for ii=1:3
                            load_x = i+ii-2 > 0 && i+ii-2<=nx;
                            if load_x & load_y & load_z
                                x_window[ii,jj,kk] = x[i+ii-2,j+jj-2,k+kk-2]
                            else
                                x_window[ii,jj,kk] = complex(0,0)
                            end
                        end
                    end
                end

                ξxM = ξx(i-0.5)
                ξxN = ξx(i)
                ξxP = ξx(i+0.5)
                ξxlo = ξxM*ξxN*hx_isq
                ξxhi = ξxP*ξxN*hx_isq
                wn2 = ω2*(v[i,j,k]^(-2))
                coef[M,N,N] = -ξxlo
                coef[P,N,N] = -ξxhi
                coef[N,N,N] = ξxlo + ξxhi + ξylo + ξyhi + ξzlo + ξzhi - wn2
                y[i,j,k] = dot(vec(coef),vec(x_window))
            end
        end
    end
    return vec(y)
end


function helm3d_std_7pt_matrix(v::Union{AbstractArray{F,3},AbstractArray{Complex{F},3}},Δ::AbstractArray{F,1},n::AbstractArray{I,1},freq::Union{F,Complex{F}},npml::AbstractArray{I,2};forw_mode::Bool=true,deriv_mode::Bool=true) where {F<:Real,I<:Integer}
    nx,ny,nz = n[1],n[2],n[3]
    hx,hy,hz = Δ[1],Δ[2],Δ[3]
    hx_isq = 1/hx^2
    hy_isq = 1/hy^2
    hz_isq = 1/hz^2
    M = 1
    N = 2
    P = 3
    ω2 = (2*π*freq)^2
    npxlo,npxhi,npylo,npyhi,npzlo,npzhi = npml[1,1],npml[2,1],npml[1,2],npml[2,2],npml[1,3],npml[2,3]
    maxnpx = max(npxlo,npxhi)
    maxnpy = max(npylo,npyhi)
    maxnpz = max(npzlo,npzhi)
    ξx = x->ξ(x,nx,npxlo,npxhi,maxnpx,freq)
    ξy = y->ξ(y,ny,npylo,npyhi,maxnpy,freq)
    ξz = z->ξ(z,nz,npzlo,npzhi,maxnpz,freq)
    coef = Array{Array{Complex{F},3},1}()
    NNM = 1
    NMN = 2
    MNN = 3
    NNN = 4
    PNN = 5
    NPN = 6
    NNP = 7

    for i=1:7
        push!(coef,zeros(Complex{F},nx,ny,nz))
    end

    for k=1:nz
        ξzM = ξz(k-0.5)
        ξzN = ξz(k)
        ξzP = ξz(k+0.5)
        ξzlo = ξzM*ξzN*hz_isq
        ξzhi = ξzP*ξzN*hz_isq
        coef[NNM][:,:,k] = -ξzlo
        coef[NNP][:,:,k] = -ξzhi
        for j=1:ny
            ξyM = ξy(j-0.5)
            ξyN = ξy(j)
            ξyP = ξy(j+0.5)
            ξylo = ξyM*ξyN*hy_isq
            ξyhi = ξyP*ξyN*hy_isq
            coef[NMN][:,j,k] = -ξylo
            coef[NPN][:,j,k] = -ξyhi
            for i=1:nx
                ξxM = ξx(i-0.5)
                ξxN = ξx(i)
                ξxP = ξx(i+0.5)
                ξxlo = ξxM*ξxN*hx_isq
                ξxhi = ξxP*ξxN*hx_isq
                wn2 = ω2*(v[i,j,k]^(-2))
                coef[MNN][i,j,k] = -ξxlo
                coef[PNN][i,j,k] = -ξxhi
                coef[NNN][i,j,k] = ξxlo + ξxhi + ξylo + ξyhi + ξzlo + ξzhi - wn2
            end
        end
    end
    coef[MNN][1,:,:] = 0
    coef[PNN][end,:,:] = 0
    coef[NMN][:,1,:] = 0
    coef[NPN][:,end,:] = 0
    coef[NNM][:,:,1] = 0
    coef[NNP][:,:,end] = 0
    Nel = nx*ny*nz
    H = spdiagm((vec(coef[NNM])[nx*ny+1:end],vec(coef[NMN])[nx+1:end],vec(coef[MNN])[2:end],vec(coef[NNN]),vec(coef[PNN][1:end-1]),vec(coef[NPN][1:end-nx]),vec(coef[NNP][1:end-nx*ny])),(-nx*ny,-nx,-1,0,1,nx,nx*ny),Nel,Nel)


end



function helm3d_chen2012_27pt(v::Union{AbstractArray{F,3},AbstractArray{Complex{F},3}},Δ::AbstractArray{F,1},n::AbstractArray{I,1},freq::Union{F,Complex{F}},npml::AbstractArray{I,2},x::AbstractArray{Complex{F},3};forw_mode::Bool=true,deriv_mode::Bool=true) where {F<:Real,I<:Integer}
    nx,ny,nz = n[1],n[2],n[3]
    hx,hy,hz = Δ[1],Δ[2],Δ[3]
    hx_isq = 1/hx^2
    hy_isq = 1/hy^2
    hz_isq = 1/hz^2
    M = 1
    N = 2
    P = 3
    λmin = minimum(vec(v))/freq
    IGmin = λmin/maximum(Δ)
    ω2 = (2*π*freq)^2
    if IGmin >= 2 && IGmin <= 3.5
        γ1 = 0.5035127
        γ2 = 0.0720630
        γ3 = 0.4244243
        w1 = 0.4058413
        w2 = 0.1966284
        w3 = 0.5979158
        w4 = -0.2003855
    elseif IGmin > 3.5 && IGmin <= 5
        γ1 = 0.7617528
        γ2 = -0.0148152
        γ3 =  0.2530624
        w1 = 0.7602512
        w2 = -0.4883334
        w3 = 1.1153920
        w4 = -0.3873097
    elseif IGmin > 5 && IGmin <= 7
        γ1 = 0.8159342
        γ2 = -0.0340791
        γ3 = 0.2181449
        w1 = 1.1330134
        w2 = -1.5191327
        w3 = 2.1033335
        w4 = 0.7172142
    elseif IGmin > 7 && IGmin <= 9
        γ1 = 0.8354262
        γ2 = -0.0394517
        γ3 = 0.2040255
        w1 = 1.7177071
        w2 = -3.2400262
        w3 = 3.8084643
        w4 = -1.2861453
    elseif IGmin > 9 && IGmin <= 10
        γ1 = 0.8432810
        γ2 = -0.0414069
        γ3 = 0.1981258
        w1 = 2.4693294
        w2 = -5.4811311
        w3 = 6.0429826
        w4 = 2.0311809
    else
        γ1 = 0.8269996
        γ2 = 4.097e-7
        γ3 = 0.1729999
        w1 = 2.9473150
        w2 = -6.8805122
        w3 = 7.4116566
        w4 = -2.4784594
    end

    npxlo,npxhi,npylo,npyhi,npzlo,npzhi = npml[1,1],npml[2,1],npml[1,2],npml[2,2],npml[1,3],npml[2,3]
    maxnpx = max(npxlo,npxhi)
    maxnpy = max(npylo,npyhi)
    maxnpz = max(npzlo,npzhi)
    ξx = x->ξ(x,nx,npxlo,npxhi,maxnpx,freq)
    ξy = y->ξ(y,ny,npylo,npyhi,maxnpy,freq)
    ξz = z->ξ(z,nz,npzlo,npzhi,maxnpz,freq)

    k_sq = zeros(eltype(v),3,3,3)
    zero_t = convert(eltype(v),0)
    coef = zeros(Complex{F},3,3,3)
    x_window = zeros(Complex{F},3,3,3)
    y = zeros(Complex{F},n...)
    for k=1:nz
        ξzM = ξz(k-1)
        ξzNminus = ξz(k-0.5)
        ξzN = ξz(k)
        ξzNplus = ξz(k+0.5)
        ξzP = ξz(k+1)
        for j=1:ny
            ξyM = ξy(k-1)
            ξyNminus = ξy(k-0.5)
            ξyN = ξy(k)
            ξyNplus = ξy(k+0.5)
            ξyP = ξy(k+1)

            for i=1:nx
                ξxM = ξx(i-1)
                ξxNminus = ξx(i-0.5)
                ξxN = ξx(i)
                ξxNplus = ξx(i+0.5)
                ξxP = ξx(i+1)

                # Load 3x3x3 windows around the current point
                for kk=1:3
                    load_z = k+kk-2 > 0 && k+kk-2<=nz;
                    for jj=1:3
                        load_y = j+jj-2 > 0 && j+jj-2<=ny;
                        for ii=1:3
                            load_x = i+ii-2 > 0 && i+ii-2<=nx;
                            if load_x & load_y & load_z
                                k_sq[ii,jj,kk] = ω2*(v[i+ii-2,j+jj-2,k+kk-2])^(-2)
                                x_window[ii,jj,kk] = x[i+ii-2,j+jj-2,k+kk-2]
                            else
                                k_sq[ii,jj,kk] = zero_t
                                x_window[ii,jj,kk] = complex(0,0)
                            end

                        end
                    end
                end

                # Coefficient expressions
                coef[M,M,M] = γ3*hx_isq*ξxNminus*ξyM*ξzM/4 + γ3*hy_isq*ξxM*ξyNminus*ξzM/4 + γ3*hz_isq*ξxM*ξyM*ξzNminus/4 + k_sq[M,M,M]*w4*ξxM*ξyM*ξzM/8
                coef[M,M,N] = γ2*hx_isq*ξxNminus*ξyM*ξzN/4 + γ2*hy_isq*ξxM*ξyNminus*ξzN/4 - γ3*hz_isq*ξxM*ξyM*ξzNminus/4 - γ3*hz_isq*ξxM*ξyM*ξzNplus/4 + k_sq[M,M,N]*w3*ξxM*ξyM*ξzN/12
                coef[M,M,P] = γ3*hx_isq*ξxNminus*ξyM*ξzP/4 + γ3*hy_isq*ξxM*ξyNminus*ξzP/4 + γ3*hz_isq*ξxM*ξyM*ξzNplus/4 + k_sq[M,M,P]*w4*ξxM*ξyM*ξzP/8
                coef[M,N,M] = γ2*hx_isq*ξxNminus*ξyN*ξzM/4 + γ2*hz_isq*ξxM*ξyN*ξzNminus/4 - γ3*hy_isq*ξxM*ξyNminus*ξzM/4 - γ3*hy_isq*ξxM*ξyNplus*ξzM/4 + k_sq[M,N,M]*w3*ξxM*ξyN*ξzM/12
                coef[M,N,N] = γ1*hx_isq*ξxNminus*ξyN*ξzN - γ2*hy_isq*ξxM*ξyNminus*ξzN/4 - γ2*hy_isq*ξxM*ξyNplus*ξzN/4 - γ2*hz_isq*ξxM*ξyN*ξzNminus/4 - γ2*hz_isq*ξxM*ξyN*ξzNplus/4 + k_sq[M,N,N]*w2*ξxM*ξyN*ξzN/6
                coef[M,N,P] = γ2*hx_isq*ξxNminus*ξyN*ξzP/4 + γ2*hz_isq*ξxM*ξyN*ξzNplus/4 - γ3*hy_isq*ξxM*ξyNminus*ξzP/4 - γ3*hy_isq*ξxM*ξyNplus*ξzP/4 + k_sq[M,N,P]*w3*ξxM*ξyN*ξzP/12
                coef[M,P,M] = γ3*hx_isq*ξxNminus*ξyP*ξzM/4 + γ3*hy_isq*ξxM*ξyNplus*ξzM/4 + γ3*hz_isq*ξxM*ξyP*ξzNminus/4 + k_sq[M,P,M]*w4*ξxM*ξyP*ξzM/8
                coef[M,P,N] = γ2*hx_isq*ξxNminus*ξyP*ξzN/4 + γ2*hy_isq*ξxM*ξyNplus*ξzN/4 - γ3*hz_isq*ξxM*ξyP*ξzNminus/4 - γ3*hz_isq*ξxM*ξyP*ξzNplus/4 + k_sq[M,P,N]*w3*ξxM*ξyP*ξzN/12
                coef[M,P,P] = γ3*hx_isq*ξxNminus*ξyP*ξzP/4 + γ3*hy_isq*ξxM*ξyNplus*ξzP/4 + γ3*hz_isq*ξxM*ξyP*ξzNplus/4 + k_sq[M,P,P]*w4*ξxM*ξyP*ξzP/8
                coef[N,M,M] = γ2*hy_isq*ξxN*ξyNminus*ξzM/4 + γ2*hz_isq*ξxN*ξyM*ξzNminus/4 - γ3*hx_isq*ξxNminus*ξyM*ξzM/4 - γ3*hx_isq*ξxNplus*ξyM*ξzM/4 + k_sq[N,M,M]*w3*ξxN*ξyM*ξzM/12
                coef[N,M,N] = γ1*hy_isq*ξxN*ξyNminus*ξzN - γ2*hx_isq*ξxNminus*ξyM*ξzN/4 - γ2*hx_isq*ξxNplus*ξyM*ξzN/4 - γ2*hz_isq*ξxN*ξyM*ξzNminus/4 - γ2*hz_isq*ξxN*ξyM*ξzNplus/4 + k_sq[N,M,N]*w2*ξxN*ξyM*ξzN/6
                coef[N,M,P] = γ2*hy_isq*ξxN*ξyNminus*ξzP/4 + γ2*hz_isq*ξxN*ξyM*ξzNplus/4 - γ3*hx_isq*ξxNminus*ξyM*ξzP/4 - γ3*hx_isq*ξxNplus*ξyM*ξzP/4 + k_sq[N,M,P]*w3*ξxN*ξyM*ξzP/12
                coef[N,N,M] = γ1*hz_isq*ξxN*ξyN*ξzNminus - γ2*hx_isq*ξxNminus*ξyN*ξzM/4 - γ2*hx_isq*ξxNplus*ξyN*ξzM/4 - γ2*hy_isq*ξxN*ξyNminus*ξzM/4 - γ2*hy_isq*ξxN*ξyNplus*ξzM/4 + k_sq[N,N,M]*w2*ξxN*ξyN*ξzM/6
                coef[N,N,N] = -γ1*hx_isq*ξxNminus*ξyN*ξzN - γ1*hx_isq*ξxNplus*ξyN*ξzN - γ1*hy_isq*ξxN*ξyNminus*ξzN - γ1*hy_isq*ξxN*ξyNplus*ξzN - γ1*hz_isq*ξxN*ξyN*ξzNminus - γ1*hz_isq*ξxN*ξyN*ξzNplus + k_sq[N,N,N]*w1*ξxN*ξyN*ξzN
                coef[N,N,P] = γ1*hz_isq*ξxN*ξyN*ξzNplus - γ2*hx_isq*ξxNminus*ξyN*ξzP/4 - γ2*hx_isq*ξxNplus*ξyN*ξzP/4 - γ2*hy_isq*ξxN*ξyNminus*ξzP/4 - γ2*hy_isq*ξxN*ξyNplus*ξzP/4 + k_sq[N,N,P]*w2*ξxN*ξyN*ξzP/6
                coef[N,P,M] = γ2*hy_isq*ξxN*ξyNplus*ξzM/4 + γ2*hz_isq*ξxN*ξyP*ξzNminus/4 - γ3*hx_isq*ξxNminus*ξyP*ξzM/4 - γ3*hx_isq*ξxNplus*ξyP*ξzM/4 + k_sq[N,P,M]*w3*ξxN*ξyP*ξzM/12
                coef[N,P,N] = γ1*hy_isq*ξxN*ξyNplus*ξzN - γ2*hx_isq*ξxNminus*ξyP*ξzN/4 - γ2*hx_isq*ξxNplus*ξyP*ξzN/4 - γ2*hz_isq*ξxN*ξyP*ξzNminus/4 - γ2*hz_isq*ξxN*ξyP*ξzNplus/4 + k_sq[N,P,N]*w2*ξxN*ξyP*ξzN/6
                coef[N,P,P] = γ2*hy_isq*ξxN*ξyNplus*ξzP/4 + γ2*hz_isq*ξxN*ξyP*ξzNplus/4 - γ3*hx_isq*ξxNminus*ξyP*ξzP/4 - γ3*hx_isq*ξxNplus*ξyP*ξzP/4 + k_sq[N,P,P]*w3*ξxN*ξyP*ξzP/12
                coef[P,M,M] = γ3*hx_isq*ξxNplus*ξyM*ξzM/4 + γ3*hy_isq*ξxP*ξyNminus*ξzM/4 + γ3*hz_isq*ξxP*ξyM*ξzNminus/4 + k_sq[P,M,M]*w4*ξxP*ξyM*ξzM/8

                coef[P,M,N] = γ2*hx_isq*ξxNplus*ξyM*ξzN/4 + γ2*hy_isq*ξxP*ξyNminus*ξzN/4 - γ3*hz_isq*ξxP*ξyM*ξzNminus/4 - γ3*hz_isq*ξxP*ξyM*ξzNplus/4 + k_sq[P,M,N]*w3*ξxP*ξyM*ξzN/12
                coef[P,M,P] = γ3*hx_isq*ξxNplus*ξyM*ξzP/4 + γ3*hy_isq*ξxP*ξyNminus*ξzP/4 + γ3*hz_isq*ξxP*ξyM*ξzNplus/4 + k_sq[P,M,P]*w4*ξxP*ξyM*ξzP/8

                coef[P,N,M] = γ2*hx_isq*ξxNplus*ξyN*ξzM/4 + γ2*hz_isq*ξxP*ξyN*ξzNminus/4 - γ3*hy_isq*ξxP*ξyNminus*ξzM/4 - γ3*hy_isq*ξxP*ξyNplus*ξzM/4 + k_sq[P,N,M]*w3*ξxP*ξyN*ξzM/12

                coef[P,N,N] = γ1*hx_isq*ξxNplus*ξyN*ξzN - γ2*hy_isq*ξxP*ξyNminus*ξzN/4 - γ2*hy_isq*ξxP*ξyNplus*ξzN/4 - γ2*hz_isq*ξxP*ξyN*ξzNminus/4 - γ2*hz_isq*ξxP*ξyN*ξzNplus/4 + k_sq[P,N,N]*w2*ξxP*ξyN*ξzN/6

                coef[P,N,P] = γ2*hx_isq*ξxNplus*ξyN*ξzP/4 + γ2*hz_isq*ξxP*ξyN*ξzNplus/4 - γ3*hy_isq*ξxP*ξyNminus*ξzP/4 - γ3*hy_isq*ξxP*ξyNplus*ξzP/4 + k_sq[P,N,P]*w3*ξxP*ξyN*ξzP/12

                coef[P,P,M] = γ3*hx_isq*ξxNplus*ξyP*ξzM/4 + γ3*hy_isq*ξxP*ξyNplus*ξzM/4 + γ3*hz_isq*ξxP*ξyP*ξzNminus/4 + k_sq[P,P,M]*w4*ξxP*ξyP*ξzM/8
                coef[P,P,N] = γ2*hx_isq*ξxNplus*ξyP*ξzN/4 + γ2*hy_isq*ξxP*ξyNplus*ξzN/4 - γ3*hz_isq*ξxP*ξyP*ξzNminus/4 - γ3*hz_isq*ξxP*ξyP*ξzNplus/4 + k_sq[P,P,N]*w3*ξxP*ξyP*ξzN/12
                coef[P,P,P] = γ3*hx_isq*ξxNplus*ξyP*ξzP/4 + γ3*hy_isq*ξxP*ξyNplus*ξzP/4 + γ3*hz_isq*ξxP*ξyP*ξzNplus/4 + k_sq[P,P,P]*w4*ξxP*ξyP*ξzP/8

                y[i,j,k] = (vec(coef).'vec(x_window))[1]
end
end
end
return vec(y)
end

function helm3d_chen2012_27pt_matrix(v::Union{AbstractArray{F,3},AbstractArray{Complex{F},3}},Δ::AbstractArray{F,1},n::AbstractArray{I,1},freq::Union{F,Complex{F}},npml::AbstractArray{I,2};deriv_mode::Bool=true) where {F<:Real,I<:Integer}
    nx,ny,nz = n[1],n[2],n[3]
    hx,hy,hz = Δ[1],Δ[2],Δ[3]
    hx_isq = 1/hx^2
    hy_isq = 1/hy^2
    hz_isq = 1/hz^2
    M = 1
    N = 2
    P = 3
    λmin = minimum(vec(v))/freq
    IGmin = λmin/maximum(Δ)
    ω2 = (2*π*freq)^2
    if IGmin >= 2 && IGmin <= 3.5
        γ1 = 0.5035127
        γ2 = 0.0720630
        γ3 = 0.4244243
        w1 = 0.4058413
        w2 = 0.1966284
        w3 = 0.5979158
        w4 = -0.2003855
    elseif IGmin > 3.5 && IGmin <= 5
        γ1 = 0.7617528
        γ2 = -0.0148152
        γ3 =  0.2530624
        w1 = 0.7602512
        w2 = -0.4883334
        w3 = 1.1153920
        w4 = -0.3873097
    elseif IGmin > 5 && IGmin <= 7
        γ1 = 0.8159342
        γ2 = -0.0340791
        γ3 = 0.2181449
        w1 = 1.1330134
        w2 = -1.5191327
        w3 = 2.1033335
        w4 = 0.7172142
    elseif IGmin > 7 && IGmin <= 9
        γ1 = 0.8354262
        γ2 = -0.0394517
        γ3 = 0.2040255
        w1 = 1.7177071
        w2 = -3.2400262
        w3 = 3.8084643
        w4 = -1.2861453
    elseif IGmin > 9 && IGmin <= 10
        γ1 = 0.8432810
        γ2 = -0.0414069
        γ3 = 0.1981258
        w1 = 2.4693294
        w2 = -5.4811311
        w3 = 6.0429826
        w4 = 2.0311809
    else
        γ1 = 0.8269996
        γ2 = 4.097e-7
        γ3 = 0.1729999
        w1 = 2.9473150
        w2 = -6.8805122
        w3 = 7.4116566
        w4 = -2.4784594
    end

    npxlo,npxhi,npylo,npyhi,npzlo,npzhi = npml[1,1],npml[2,1],npml[1,2],npml[2,2],npml[1,3],npml[2,3]
    maxnpx = max(npxlo,npxhi)
    maxnpy = max(npylo,npyhi)
    maxnpz = max(npzlo,npzhi)
    ξx = x->ξ(x,nx,npxlo,npxhi,maxnpx,freq)
    ξy = y->ξ(y,ny,npylo,npyhi,maxnpy,freq)
    ξz = z->ξ(z,nz,npzlo,npzhi,maxnpz,freq)

    k_sq = zeros(eltype(v),3,3,3)
    zero_t = convert(eltype(v),0)
    coef = zeros(Complex{F},3,3,3,n[1],n[2],n[3])
    x_window = zeros(Complex{F},3,3,3)

    for k=1:nz
        ξzM = ξz(k-1)
        ξzNminus = ξz(k-0.5)
        ξzN = ξz(k)
        ξzNplus = ξz(k+0.5)
        ξzP = ξz(k+1)
        for j=1:ny
            ξyM = ξy(k-1)
            ξyNminus = ξy(k-0.5)
            ξyN = ξy(k)
            ξyNplus = ξy(k+0.5)
            ξyP = ξy(k+1)
            for i=1:nx
                ξxM = ξx(i-1)
                ξxNminus = ξx(i-0.5)
                ξxN = ξx(i)
                ξxNplus = ξx(i+0.5)
                ξxP = ξx(i+1)

                # Load 3x3x3 windows around the current point
                for kk=1:3
                    load_z = k+kk-2 > 0 && k+kk-2<=nz;
                    for jj=1:3
                        load_y = j+jj-2 > 0 && j+jj-2<=ny;
                        for ii=1:3
                            load_x = i+ii-2 > 0 && i+ii-2<=nx;
                            if load_x & load_y & load_z
                                k_sq[ii,jj,kk] = ω2*(v[i+ii-2,j+jj-2,k+kk-2]^(-2))
                            else
                                k_sq[ii,jj,kk] = zero_t
                            end
                        end
                    end
                end

                # Coefficient expressions
                coef[M,M,M,i,j,k] = γ3*hx_isq*ξxNminus*ξyM*ξzM/4 + γ3*hy_isq*ξxM*ξyNminus*ξzM/4 + γ3*hz_isq*ξxM*ξyM*ξzNminus/4 + k_sq[M,M,M]*w4*ξxM*ξyM*ξzM/8
                coef[M,M,N,i,j,k] = γ2*hx_isq*ξxNminus*ξyM*ξzN/4 + γ2*hy_isq*ξxM*ξyNminus*ξzN/4 - γ3*hz_isq*ξxM*ξyM*ξzNminus/4 - γ3*hz_isq*ξxM*ξyM*ξzNplus/4 + k_sq[M,M,N]*w3*ξxM*ξyM*ξzN/12
                coef[M,M,P,i,j,k] = γ3*hx_isq*ξxNminus*ξyM*ξzP/4 + γ3*hy_isq*ξxM*ξyNminus*ξzP/4 + γ3*hz_isq*ξxM*ξyM*ξzNplus/4 + k_sq[M,M,P]*w4*ξxM*ξyM*ξzP/8
                coef[M,N,M,i,j,k] = γ2*hx_isq*ξxNminus*ξyN*ξzM/4 + γ2*hz_isq*ξxM*ξyN*ξzNminus/4 - γ3*hy_isq*ξxM*ξyNminus*ξzM/4 - γ3*hy_isq*ξxM*ξyNplus*ξzM/4 + k_sq[M,N,M]*w3*ξxM*ξyN*ξzM/12
                coef[M,N,N,i,j,k] = γ1*hx_isq*ξxNminus*ξyN*ξzN - γ2*hy_isq*ξxM*ξyNminus*ξzN/4 - γ2*hy_isq*ξxM*ξyNplus*ξzN/4 - γ2*hz_isq*ξxM*ξyN*ξzNminus/4 - γ2*hz_isq*ξxM*ξyN*ξzNplus/4 + k_sq[M,N,N]*w2*ξxM*ξyN*ξzN/6
                coef[M,N,P,i,j,k] = γ2*hx_isq*ξxNminus*ξyN*ξzP/4 + γ2*hz_isq*ξxM*ξyN*ξzNplus/4 - γ3*hy_isq*ξxM*ξyNminus*ξzP/4 - γ3*hy_isq*ξxM*ξyNplus*ξzP/4 + k_sq[M,N,P]*w3*ξxM*ξyN*ξzP/12
                coef[M,P,M,i,j,k] = γ3*hx_isq*ξxNminus*ξyP*ξzM/4 + γ3*hy_isq*ξxM*ξyNplus*ξzM/4 + γ3*hz_isq*ξxM*ξyP*ξzNminus/4 + k_sq[M,P,M]*w4*ξxM*ξyP*ξzM/8
                coef[M,P,N,i,j,k] = γ2*hx_isq*ξxNminus*ξyP*ξzN/4 + γ2*hy_isq*ξxM*ξyNplus*ξzN/4 - γ3*hz_isq*ξxM*ξyP*ξzNminus/4 - γ3*hz_isq*ξxM*ξyP*ξzNplus/4 + k_sq[M,P,N]*w3*ξxM*ξyP*ξzN/12
                coef[M,P,P,i,j,k] = γ3*hx_isq*ξxNminus*ξyP*ξzP/4 + γ3*hy_isq*ξxM*ξyNplus*ξzP/4 + γ3*hz_isq*ξxM*ξyP*ξzNplus/4 + k_sq[M,P,P]*w4*ξxM*ξyP*ξzP/8
                coef[N,M,M,i,j,k] = γ2*hy_isq*ξxN*ξyNminus*ξzM/4 + γ2*hz_isq*ξxN*ξyM*ξzNminus/4 - γ3*hx_isq*ξxNminus*ξyM*ξzM/4 - γ3*hx_isq*ξxNplus*ξyM*ξzM/4 + k_sq[N,M,M]*w3*ξxN*ξyM*ξzM/12
                coef[N,M,N,i,j,k] = γ1*hy_isq*ξxN*ξyNminus*ξzN - γ2*hx_isq*ξxNminus*ξyM*ξzN/4 - γ2*hx_isq*ξxNplus*ξyM*ξzN/4 - γ2*hz_isq*ξxN*ξyM*ξzNminus/4 - γ2*hz_isq*ξxN*ξyM*ξzNplus/4 + k_sq[N,M,N]*w2*ξxN*ξyM*ξzN/6
                coef[N,M,P,i,j,k] = γ2*hy_isq*ξxN*ξyNminus*ξzP/4 + γ2*hz_isq*ξxN*ξyM*ξzNplus/4 - γ3*hx_isq*ξxNminus*ξyM*ξzP/4 - γ3*hx_isq*ξxNplus*ξyM*ξzP/4 + k_sq[N,M,P]*w3*ξxN*ξyM*ξzP/12
                coef[N,N,M,i,j,k] = γ1*hz_isq*ξxN*ξyN*ξzNminus - γ2*hx_isq*ξxNminus*ξyN*ξzM/4 - γ2*hx_isq*ξxNplus*ξyN*ξzM/4 - γ2*hy_isq*ξxN*ξyNminus*ξzM/4 - γ2*hy_isq*ξxN*ξyNplus*ξzM/4 + k_sq[N,N,M]*w2*ξxN*ξyN*ξzM/6
                coef[N,N,N,i,j,k] = -γ1*hx_isq*ξxNminus*ξyN*ξzN - γ1*hx_isq*ξxNplus*ξyN*ξzN - γ1*hy_isq*ξxN*ξyNminus*ξzN - γ1*hy_isq*ξxN*ξyNplus*ξzN - γ1*hz_isq*ξxN*ξyN*ξzNminus - γ1*hz_isq*ξxN*ξyN*ξzNplus + k_sq[N,N,N]*w1*ξxN*ξyN*ξzN
                coef[N,N,P,i,j,k] = γ1*hz_isq*ξxN*ξyN*ξzNplus - γ2*hx_isq*ξxNminus*ξyN*ξzP/4 - γ2*hx_isq*ξxNplus*ξyN*ξzP/4 - γ2*hy_isq*ξxN*ξyNminus*ξzP/4 - γ2*hy_isq*ξxN*ξyNplus*ξzP/4 + k_sq[N,N,P]*w2*ξxN*ξyN*ξzP/6
                coef[N,P,M,i,j,k] = γ2*hy_isq*ξxN*ξyNplus*ξzM/4 + γ2*hz_isq*ξxN*ξyP*ξzNminus/4 - γ3*hx_isq*ξxNminus*ξyP*ξzM/4 - γ3*hx_isq*ξxNplus*ξyP*ξzM/4 + k_sq[N,P,M]*w3*ξxN*ξyP*ξzM/12
                coef[N,P,N,i,j,k] = γ1*hy_isq*ξxN*ξyNplus*ξzN - γ2*hx_isq*ξxNminus*ξyP*ξzN/4 - γ2*hx_isq*ξxNplus*ξyP*ξzN/4 - γ2*hz_isq*ξxN*ξyP*ξzNminus/4 - γ2*hz_isq*ξxN*ξyP*ξzNplus/4 + k_sq[N,P,N]*w2*ξxN*ξyP*ξzN/6
                coef[N,P,P,i,j,k] = γ2*hy_isq*ξxN*ξyNplus*ξzP/4 + γ2*hz_isq*ξxN*ξyP*ξzNplus/4 - γ3*hx_isq*ξxNminus*ξyP*ξzP/4 - γ3*hx_isq*ξxNplus*ξyP*ξzP/4 + k_sq[N,P,P]*w3*ξxN*ξyP*ξzP/12
                coef[P,M,M,i,j,k] = γ3*hx_isq*ξxNplus*ξyM*ξzM/4 + γ3*hy_isq*ξxP*ξyNminus*ξzM/4 + γ3*hz_isq*ξxP*ξyM*ξzNminus/4 + k_sq[P,M,M]*w4*ξxP*ξyM*ξzM/8

                coef[P,M,N,i,j,k] = γ2*hx_isq*ξxNplus*ξyM*ξzN/4 + γ2*hy_isq*ξxP*ξyNminus*ξzN/4 - γ3*hz_isq*ξxP*ξyM*ξzNminus/4 - γ3*hz_isq*ξxP*ξyM*ξzNplus/4 + k_sq[P,M,N]*w3*ξxP*ξyM*ξzN/12
                coef[P,M,P,i,j,k] = γ3*hx_isq*ξxNplus*ξyM*ξzP/4 + γ3*hy_isq*ξxP*ξyNminus*ξzP/4 + γ3*hz_isq*ξxP*ξyM*ξzNplus/4 + k_sq[P,M,P]*w4*ξxP*ξyM*ξzP/8

                coef[P,N,M,i,j,k] = γ2*hx_isq*ξxNplus*ξyN*ξzM/4 + γ2*hz_isq*ξxP*ξyN*ξzNminus/4 - γ3*hy_isq*ξxP*ξyNminus*ξzM/4 - γ3*hy_isq*ξxP*ξyNplus*ξzM/4 + k_sq[P,N,M]*w3*ξxP*ξyN*ξzM/12

coef[P,N,N,i,j,k] = γ1*hx_isq*ξxNplus*ξyN*ξzN - γ2*hy_isq*ξxP*ξyNminus*ξzN/4 - γ2*hy_isq*ξxP*ξyNplus*ξzN/4 - γ2*hz_isq*ξxP*ξyN*ξzNminus/4 - γ2*hz_isq*ξxP*ξyN*ξzNplus/4 + k_sq[P,N,N]*w2*ξxP*ξyN*ξzN/6

                coef[P,N,P,i,j,k] = γ2*hx_isq*ξxNplus*ξyN*ξzP/4 + γ2*hz_isq*ξxP*ξyN*ξzNplus/4 - γ3*hy_isq*ξxP*ξyNminus*ξzP/4 - γ3*hy_isq*ξxP*ξyNplus*ξzP/4 + k_sq[P,N,P]*w3*ξxP*ξyN*ξzP/12

                coef[P,P,M,i,j,k] = γ3*hx_isq*ξxNplus*ξyP*ξzM/4 + γ3*hy_isq*ξxP*ξyNplus*ξzM/4 + γ3*hz_isq*ξxP*ξyP*ξzNminus/4 + k_sq[P,P,M]*w4*ξxP*ξyP*ξzM/8
                coef[P,P,N,i,j,k] = γ2*hx_isq*ξxNplus*ξyP*ξzN/4 + γ2*hy_isq*ξxP*ξyNplus*ξzN/4 - γ3*hz_isq*ξxP*ξyP*ξzNminus/4 - γ3*hz_isq*ξxP*ξyP*ξzNplus/4 + k_sq[P,P,N]*w3*ξxP*ξyP*ξzN/12
                coef[P,P,P,i,j,k] = γ3*hx_isq*ξxNplus*ξyP*ξzP/4 + γ3*hy_isq*ξxP*ξyNplus*ξzP/4 + γ3*hz_isq*ξxP*ξyP*ξzNplus/4 + k_sq[P,P,P]*w4*ξxP*ξyP*ξzP/8


end
end
end


# Zero out coefficients on the boundary layers
for i=[M N P]
    for j=[M N P]
        for k=[M N P]
            if i==M
                coef[i,j,k,1,:,:] = 0
            elseif i==P
                coef[i,j,k,end,:,:] = 0
            end
            if j==M
                coef[i,j,k,:,1,:] = 0
            elseif j==P
                coef[i,j,k,:,end,:] = 0
            end
            if k==M
                coef[i,j,k,:,:,1] = 0
            elseif k==P
                coef[i,j,k,:,:,end] = 0
            end
        end
    end
end


coef_tup = Array{Array{Complex{F},1},1}()
offset_tup = Array{I,1}()
Nel = nx*ny*nz
t = 1
for k=[M N P]
    for j=[M N P]
        for i=[M N P]
            idx_start = 1
            idx_end = Nel
            offset = 0
            # For the coefficients before the diagonal, in lexographial organization
            # Remove the coefficient
            if t < 14
                if i==M
                    idx_start += 1
                    offset -= 1
                elseif i==P
                    idx_start -= 1
                    offset += 1
                end
                if j==M
                    idx_start += nx
                    offset -= nx
                elseif j==P
                    idx_start -= nx
                    offset += nx
                end
                if k==M
                    idx_start += nx*ny
                    offset -= nx*ny
                end
            elseif t > 14
                if i==M
                    idx_end += 1
                    offset -= 1
                elseif i==P
                    idx_end -= 1
                    offset += 1
                end
                if j==M
                    idx_end += nx
                    offset -= nx
                elseif j==P
                    idx_end -= nx
                    offset += nx
                end
                if k==P
                    idx_end -= nx*ny
                    offset += nx*ny
                end
            end
            push!(coef_tup,vec(coef[i,j,k,:,:,:])[idx_start:idx_end])
            push!(offset_tup,offset)

            t += 1
        end
    end
end

H = spdiagm(tuple(coef_tup...),tuple(offset_tup...),Nel,Nel)

end


function ξ(x,nx,nplo,nphi,maxnp,freq)
    C = 10;
    dist_from_int = (x .< (nplo+1)) .* (nplo+1-x) + (x .>= nx-nphi) .* (x-(nx-nphi));
    sigma = C * (dist_from_int/maxnp).^2;
    func = complex(1.0,sigma/freq);
end


function pml_func(nx::Int,np_lo::Int,np_hi::Int)

    s_hx = 1.0/(nx+1)
    xix = zeros(nx,2)
    Lx_hi = np_hi/nx
    Lx_lo = np_lo/nx
    gamma = zeros(nx+2,1)
    gamma[1:np_lo] = cos.((pi*(0:np_lo-1) * s_hx)/(2*Lx_lo))
    gamma[np_lo+1:nx+2-np_hi] = 0
    gamma[nx+3-np_hi:nx+2] = cos.((pi*(1-(nx+2-np_hi:nx+1)*s_hx))/(2*Lx_hi))
    return 2.0./((1 + im*gamma[2:nx+1]).*(1 + im*gamma[2:nx+1] + 1 + im*gamma[1:nx] )), 2.0./((1 + im*gamma[2:nx+1]).*(1 + im*gamma[2:nx+1] + 1 + im*gamma[3:nx+2] ))

end
