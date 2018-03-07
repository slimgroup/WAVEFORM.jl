using Revise
using Waveform

nx = 70
npx = 10
n = nx*ones(Int64,3)
w = randn(n...);
x = randn(n...) + im*randn(n...);
Δ = [1.0; 1.0; 1.0]
n = [70; 70; 70]
freq = 4.0
npml = 10*ones(Int,2,3)

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


function helm3d_operto_mvp_forw_impl!(wn, x, y, n, params, deriv_mode, z_idx) 
    M, N, P = 1, 2, 3
    F = Float64
    CF = Complex{Float64}
    wn_window = zeros(eltype(wn),3,3,3)
    coef = zeros(Complex{Float64},3,3,3)
    x_window = zeros(Complex{Float64},3,3,3)
    zero_x = complex(0.0,0.0)
    zero_w = zero(eltype(wn))
    nx,ny,nz = n
    is_deriv_mode = deriv_mode ? complex(0.0,0.0) : complex(1.0,0.0)
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


function helm3d_operto_mvp_mt(wn,Δ,n,freq,npml,x)
    forw_mode = true
    deriv_mode = false
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
    num_blocks = Threads.nthreads()
    block_size = ceil(Int64,nz/num_blocks)::Int64
    Threads.@threads for i=1:num_blocks
        z_idx = (i-1)*block_size+1:min(nz,i*block_size)
        helm3d_operto_mvp_forw_impl!(wn, x, y, n, params, deriv_mode, z_idx)
    end
    return vec(y)
end



function helm3d_operto_mvp(wn,Δ,n,freq,npml,x)
    forw_mode = true
    deriv_mode = false
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
        w3a = 2*(w3*3/(4*hxyz))::Float64

        cx = - (w1/hx + w2/hx + w2/hxz + w2/hxy + 4*w3a)
        cy = - (w1/hy + w2/hy + w2/hyz + w2/hxy + 4*w3a)
        cz = - (w1/hz + w2/hz + w2/hxz + w2/hyz + 4*w3a)
        cNNN = - (w1 + 3*w2 + (8*w3a*hxyz)/3 + wm1-1)

        xz_coef = w2/(2*hxz)
        xy_coef = w2/(2*hxy)
        yz_coef = w2/(2*hyz)
        wn_window = zeros(Float64,3,3,3)
        coef = zeros(Complex{Float64},3,3,3)
        x_window = zeros(Complex{Float64},3,3,3)
        zero_x = complex(0.0,0.0)
        zero_w = zero(eltype(wn))
        is_deriv_mode = deriv_mode ? complex(0.0,0.0) : complex(1.0,0.0)
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
    return y
    end