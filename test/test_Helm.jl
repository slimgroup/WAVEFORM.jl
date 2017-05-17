tsname="Helm"
T = 3
@testset "$tsname" begin
    n = 100*[1;1];
    d = 50.0*[1;1];
    o = 0.0*[1;1];
    v0 = 2000;
    
    t0 = 0.0;
    f0 = 10.0;
    unit = "m/s";
    xsrc = linspace(0.0,(n.*d)[2]::Float64,n[1]);
    ysrc = [0.0];
    zsrc = [100.0];
    xrec = linspace(0.0,(n.*d)[2]::Float64,n[2]);
    yrec = [0.0];
    zrec = [100.0];
    freqs = [4.0];
    λ = v0/freqs[1]
    
    model = Waveform.Model{Int64,Float64}(n,d,o,t0,f0,unit,freqs,xsrc,ysrc,zsrc,xrec,yrec,zrec)
    nsrc = length(model.xsrc);
    
    comp_n = n;
    comp_d = d;
    comp_o = o;
    npml = convert(Int,λ/minimum(comp_d))*[1 1; 1 1];
    scheme = Waveform.helm2d_chen9p;
    cut_pml = true;
    misfit = Waveform.least_squares;
    opts = Waveform.PDEopts{Int64,Float64}(scheme,comp_n,comp_d,comp_o,cut_pml,npml,misfit)
    
    v = v0*ones(n...);
    v = vec(v);
    v1 = v0*ones(n...);
    v1[div(n[1],4):3*div(n[1],4),div(n[2],4):3*div(n[2],4)] = 1.5*v0;
    v1 = vec(v1);

    Q = eye(nsrc);


    (H,comp_grid,T_forw,T_adj,DT_adj) = Waveform.discrete_helmholtz(v,model,freqs[1],opts);
    
    (zt,xt) = Waveform.odn_to_grid(comp_grid.comp_o,comp_grid.comp_d,comp_grid.comp_n);

    Psrcz = joSincInterp(model.zsrc,zt,r=4);
    Psrcx = joSincInterp(model.xsrc,xt,r=4);
    Psrc = joKron(Psrcx,Psrcz);
    
    q = jo_convert(Complex{Float64},Psrc*Q[:,div(nsrc,2)]);
    u = H\q;
    @test norm(H*u-q)<1e-8*norm(q)
    error_exponent = x->median(log10(x[2:end]./x[1:end-1]))
    dot_test(x,y) = abs(real(x)-real(y))<1e-10*max(abs(real(x)),abs(real(y)))
    for j in 1:T
        dv = randn(n...);
        dv[[1 end],:] = 0.0;
        dv[:,[1 end]] = 0.0;
        dv = vec(dv);
        h = 10.0.^(-6:0);
        e0 = zeros(length(h));
        e1 = zeros(length(h));
        Hu = H*u;
        dHu = T_forw(u,comp_grid.phys_to_comp_grid*dv);
        for i in 1:length(h)
            (H1,c1,T1,T2,T3) = Waveform.discrete_helmholtz(v+h[i]*dv,model,freqs[1],opts);
            H1u = H1*u;
            e0[i] = norm(H1u-Hu);
            e1[i] = norm(H1u-Hu-h[i]*dHu);
        end
        
        @test abs(error_exponent(e0)-1)<0.05
        @test abs(error_exponent(e1)-2)<0.05
        z = randn(size(dHu)) + im*randn(size(dHu))
        s = dot(z,dHu)
        t = dot(T_adj(u,z),comp_grid.phys_to_comp_grid*dv)
        @test dot_test(s,t)

        # Test forward modeling error
        D = Waveform.PDEfunc(Waveform.forw_model,v,vec(Q),Array(Float64,0),Array(Float64,0),model,opts);
        δD = Waveform.PDEfunc(Waveform.jacob_forw,v,vec(Q),Array(Float64,0),dv,model,opts);
        for i in 1:length(h)
            D1 = Waveform.PDEfunc(Waveform.forw_model,v+h[i]*dv,vec(Q),Array(Float64,0),Array(Float64,0),model,opts);
            e0[i] = vecnorm(D1-D);
            e1[i] = vecnorm(D1-D-h[i]*δD);
        end
        @test abs(error_exponent(e0)-1)<0.2
        @test abs(error_exponent(e1)-2)<0.2

        # Test adjoint 
        Z = randn(size(δD))+im*randn(size(δD));
        s = Waveform.PDEfunc(Waveform.jacob_adj,v,vec(Q),Array(Float64,0),vec(Z),model,opts)
        
        @test dot_test(dot(vec(δD),vec(Z)), dot(s,dv))
        
        (f,g) = Waveform.PDEfunc(Waveform.objective,v1,vec(Q),vec(D),Array(Float64,0),model,opts,compute_grad=true)
        df = dot(vec(dv),g);
        for i in 1:length(h)
            f1 = Waveform.PDEfunc(Waveform.objective,v1+h[i]*dv,vec(Q),vec(D),Array(Float64,0),model,opts)
            e0[i] = abs(f1-f)
            e1[i] = abs(f1-f-h[i]*df)
        end
        @test abs(error_exponent(e0)-1)<0.2
        @test abs(error_exponent(e1)-2)<0.2
        
    end
end
