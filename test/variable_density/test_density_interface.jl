@testset "Density-only interface: refinement against a fine reference" begin
    # constant c, density jump 1000 -> 2500 kg/m³ at z = 300 m (nodes on the interface take the
    # lower value). Spec Sec. 4: smooth-coefficient order does not carry over to sharp jumps.
    f, c0, L, Lp = 10.0, 2000.0, 600.0, 200.0
    ω = 2π * f
    ρfun(z) = z < 300 ? 1000.0 : 2500.0
    xr = collect(100.0:10.0:500.0)
    acq = Acquisition([150.0], [300.0], vcat(fill(100.0, length(xr)), fill(500.0, length(xr))), vcat(xr, xr))
    function data(h)
        n, np_ = round(Int, L / h) + 1, round(Int, Lp / h)
        g = Grid2D(n, n, h, 0.0, 0.0, (np_, np_, np_, np_))
        zs = [comp_z(g, i) for i in physical_z_range(g)]
        m = Material(fill(c0, n, n), [ρfun(z) for z in zs, _ in 1:n])
        return predict_data(HelmholtzPML(g, m, ω, CHEN2013_GLOBAL, chen_stretch(g, ω; f0=10.0)), acq, [1.0])
    end
    dref = data(1.25)
    errs = [norm(data(h) - dref) / norm(dref) for h in (10.0, 5.0, 2.5)]
    rates = [log2(errs[i] / errs[i+1]) for i in 1:2]
    @info "density interface" errs rates
    @test all(rates .>= 1.0)                 # measured 1.23, 1.59
    @test errs[end] < 1e-2                   # measured 6.8e-3
end
