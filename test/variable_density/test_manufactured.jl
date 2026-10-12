@testset "Manufactured solution: variable c, ρ through the PML" begin
    Lz, Lx, Lp = 600.0, 800.0, 160.0
    ω, f0, a0 = 2π * 6.0, 10.0, 1.79
    cphys(z, x) = 2000 + 300 * cos(2π * z / Lz) * cos(π * x / Lx)      # zero normal derivative on the boundary
    rphys(z, x) = 1500 + 400 * cos(π * z / Lz) * cos(2π * x / Lx)
    cfun(z, x) = cphys(clamp(z, 0, Lz), clamp(x, 0, Lx))               # = extend_to_pml
    rfun(z, x) = rphys(clamp(z, 0, Lz), clamp(x, 0, Lx))
    σ(l) = 2π * a0 * f0 * (l / Lp)^2                                    # = chen_stretch, physical units
    sfun(t, L) = t < 0 ? 1 + im * σ(-t) / ω : (t > L ? 1 + im * σ(t - L) / ω : complex(1.0))
    sz(z) = sfun(z, Lz); sx(x) = sfun(x, Lx)
    bump(r2) = r2 < 0.98 ? exp(-1 / (1 - r2)) : zero(r2)               # exp(-50) ≈ 2e-22 at the cut-off
    ustar(z, x) = bump(((z - 100)^2 + (x - 400)^2) / 240^2) * cis(0.012 * x + 0.007 * z)
    D(fun, t) = ForwardDiff.derivative(s -> real(fun(s)), t) + im * ForwardDiff.derivative(s -> imag(fun(s)), t)
    function frhs(z, x)
        a = 1 / (rfun(z, x) * cfun(z, x)^2)
        fx = D(xx -> (1 / rfun(z, xx)) * sz(z) / sx(xx) * D(t -> ustar(z, t), xx), x)
        fz = D(zz -> (1 / rfun(zz, x)) * sx(x) / sz(zz) * D(t -> ustar(t, x), zz), z)
        return -fx - fz - ω^2 * a * sx(x) * sz(z) * ustar(z, x)
    end
    hs = [10.0, 5.0, 2.5]
    err_trunc = Float64[]; err_sol = Float64[]
    for h in hs
        nz, nx, np_ = round(Int, Lz / h) + 1, round(Int, Lx / h) + 1, round(Int, Lp / h)
        g = Grid2D(nz, nx, h, 0.0, 0.0, (np_, np_, np_, np_))
        zs = [comp_z(g, i) for i in 1:comp_size(g)[1]]; xs = [comp_x(g, j) for j in 1:comp_size(g)[2]]
        zp, xp = zs[physical_z_range(g)], xs[physical_x_range(g)]
        m = Material([cphys(z, x) for z in zp, x in xp], [rphys(z, x) for z in zp, x in xp])
        op = HelmholtzPML(g, m, ω, CHEN2013_GLOBAL, chen_stretch(g, ω; f0=f0, a0=a0))
        U = [ustar(z, x) for z in zs, x in xs]
        F = [frhs(z, x) for z in zs, x in xs]
        T = reference_apply(op, U) - F
        push!(err_trunc, maximum(abs, T[3:end-2, 3:end-2]) / maximum(abs, F))
        uh = reshape(lu(assemble_sparse(op)) \ vec(F), size(U))
        push!(err_sol, norm(uh - U) / norm(U))
    end
    rate(e) = [log2(e[i] / e[i+1]) for i in 1:length(e)-1]
    @info "manufactured solution" err_sol rate(err_sol) err_trunc rate(err_trunc)
    @test all(1.8 .<= rate(err_sol) .<= 2.3)                  # measured 2.04, 2.00
    @test issorted(err_trunc, rev=true) && rate(err_trunc)[end] >= 1.4   # measured 1.57, 1.78
end
