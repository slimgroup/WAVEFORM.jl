@testset "2D Helmholtz" begin
    n = 101*[1;1]; d = 10.0*[1;1]; o = 0.0*[1;1]; L = n.*d

    @testset "Green's function, f=$f Hz" for f in [5.0, 10.0, 15.0]
        # point source at (z,x) = (500,500), receivers along z = 200
        model = Model{Int64,Float64}(n,d,o,0.0,10.0,"m/s",[f],[500.0],[0.0],[500.0],collect(0.0:10.0:L[2]),[0.0],[200.0])
        opts = PDEopts{Int64,Float64}(helm2d_chen9p,n,d,o,true,false,50*ones(Int64,2,2),least_squares,trues(1,1),LinSolveOpts(solver=:lufact))
        D = vec(forw_model(2000*ones(prod(n)),ones(1,1),model,opts))
        r = hypot.(model.xrec .- 500.0, 300.0)
        # WAVEFORM uses the exp(+iωt) convention, so the outgoing solution is conj((i/4) H0(kr))
        G = conj.((im/4)*hankelh1.(0,2π*f/2000*r))
        c = (D'*G)/(D'*D)  # unknown complex scale (wavelet, normalisation)
        @test norm(c*D-G)/norm(G) < 1e-2
    end

    freqs = [5.0;10.0]
    model = Model{Int64,Float64}(n,d,o,0.0,10.0,"m/s",freqs,collect(0.0:100.0:L[2]),[0.0],[10.0],collect(0.0:10.0:L[2]),[0.0],[950.0])
    nsrc = length(model.xsrc); nfreq = length(freqs); nrec = length(model.xrec)
    Q = Matrix{Float64}(I,nsrc,nsrc)
    v = 2000*ones(n...); v[34:67,34:67] .= 2500; v = vec(v)
    v0 = 2000*ones(prod(n))

    @testset "Jacobian adjoint test" begin
        opts = PDEopts{Int64,Float64}(helm2d_chen9p,n,d,o,false,false,20*ones(Int64,2,2),least_squares,trues(nsrc,nfreq),LinSolveOpts(solver=:lufact))
        δm = randn(prod(n))
        δd = randn(ComplexF64,nrec*nsrc*nfreq)
        Jδm = PDEfunc!(:jacob_forw,v0,vec(Q),Float64[],δm,model,opts)
        JTδd = PDEfunc!(:jacob_adj,v0,vec(Q),Float64[],δd,model,opts)
        @test abs(real(dot(δd,vec(Jδm))) - dot(JTδd,δm)) < 1e-10*abs(dot(JTδd,δm))
    end

    @testset "Misfit gradient Taylor test" begin
        # cut_pml=false: the gradient includes the PML-zone sensitivity of the boundary cells
        opts = PDEopts{Int64,Float64}(helm2d_chen9p,n,d,o,false,false,20*ones(Int64,2,2),least_squares,trues(nsrc,nfreq),LinSolveOpts(solver=:lufact))
        D = forw_model(v,Q,model,opts)
        obj! = construct_pde_misfit(v,Q,D,model,opts,batch_mode=false)
        g = zeros(prod(n)); f0 = obj!(v0,g)
        δv = 10*randn(prod(n))
        h = 10.0 .^ (-1:-1:-4)
        err0 = [abs(obj!(v0 .+ hi*δv, zeros(prod(n))) - f0) for hi in h]
        err1 = [abs(obj!(v0 .+ hi*δv, zeros(prod(n))) - f0 - hi*dot(g,δv)) for hi in h]
        @test abs(convergence_rate(h,err0) - 1) < 0.1
        @test abs(convergence_rate(h,err1) - 2) < 0.1
    end
end
