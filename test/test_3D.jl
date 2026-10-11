@testset "3D Helmholtz" begin
    n = 40*[1;1;1]; d = 12.5*[1;1;1]; o = 0.0*[1;1;1]
    v0 = 2000.0; freq = 8.0
    model = Model{Int64,Float64}(n,d,o,0.0,10.0,"m/s",[freq],[(n.*d)[1]/2],[(n.*d)[2]/2],[100.0],
                                 collect(range(0.0,(n.*d)[1],length=n[1])),collect(range(0.0,(n.*d)[2],length=n[2])),[100.0])
    lsopts = LinSolveOpts(solver=:fgmres,maxinnerit=5,tol=1e-8)
    lsopts.precond = :mlgmres
    opts = PDEopts{Int64,Float64}(helm3d_operto27,n,d,o,true,true,10*ones(Int64,2,3),least_squares,trues(1,1),lsopts)
    v = v0*ones(prod(n))

    (H,comp_grid,T,DT_adj,P) = helmholtz_system(v,model,freq,opts)
    nt = comp_grid.comp_n
    q = zeros(ComplexF64,nt...); src = div.(nt,2); q[src...] = 1.0; q = vec(q)

    @testset "Solves" begin
        u = H\q
        @test norm(H*u-q) < 1e-7*norm(q)
        ut = H'\q
        @test norm(H'*ut-q) < 1e-7*norm(q)
    end

    @testset "Green's function" begin
        u = reshape(H\q,nt...)
        (xt,yt,zt) = odn_to_grid(comp_grid)
        k = 2π*freq/v0
        r = [sqrt((x-xt[src[1]])^2 + (y-yt[src[2]])^2 + (z-zt[src[3]])^2) for x in xt, y in yt, z in zt]
        # compare inside the physical domain (outside the PML), away from the source singularity
        npml = div.(nt .- comp_grid.comp_n_nopml, 2)
        mask = falses(nt...)
        mask[(npml[1]+1:nt[1]-npml[1]),(npml[2]+1:nt[2]-npml[2]),(npml[3]+1:nt[3]-npml[3])] .= true
        mask .&= r .> 3*d[1]
        # Unlike the 2D stencils, the 3D 27-point operator uses the exp(-iωt) convention,
        # so the outgoing solution is exp(+ikr)/(4πr)
        G = exp.(im*k*r[mask])./(4π*r[mask])
        um = u[mask]
        c = (um'*G)/(um'*um)
        @test norm(c*um-G)/norm(G) < 0.05
    end

    @testset "Wavefield Taylor test" begin
        u0 = H\q
        δv = randn(prod(n))
        δu = H\(-T(u0)*(comp_grid.phys_to_comp_grid*δv))
        h = 10.0 .^ (0:-1:-3)
        err0 = zeros(length(h)); err1 = zeros(length(h))
        for (i,hi) in enumerate(h)
            u1 = helmholtz_system(v+hi*δv,model,freq,opts)[1]\q
            err0[i] = norm(u1-u0)
            err1[i] = norm(u1-u0-hi*δu)
        end
        @test abs(convergence_rate(h,err0) - 1) < 0.2
        @test abs(convergence_rate(h,err1) - 2) < 0.2
    end
end
