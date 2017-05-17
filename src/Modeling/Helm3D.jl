




function helm3d_chen2012_27pt!{F<:Real,I<:Integer}(v::Union{AbstractArray{F,3},AbstractArray{Complex{F},3}},Δ::AbstractArray{F,1},n::AbstractArray{I,1},freq::Union{F,Complex{F}},npml::AbstractArray{I,2},x::AbstractArray{Complex{F},3},y::AbstractArray{Complex{F},3};forw_mode::Bool=true,deriv_mode::Bool=true)
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
                ξxM = ξx(k-1)
                ξxNminus = ξx(k-0.5)
                ξxN = ξx(k)
                ξxNplus = ξx(k+0.5)
                ξxP = ξx(k+1)
                
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

                y[i,j,k] = dot(vec(coef),vec(x_window))

            end
        end
    end
end

function ξ(x,nx,nplo,nphi,maxnp,freq) 
    C = 10;
    dist_from_int = (x <= nplo) .* (nplo-x) + (x > nx-nphi) .* (x-(nx-nphi+1));
    sigma = C * (dist_from_int/maxnp).^2;
    func = complex(1.0,-sigma/freq);
end
