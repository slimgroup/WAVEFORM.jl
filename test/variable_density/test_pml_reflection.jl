@testset "PML reflection vs thickness (incl. corner)" begin
    # Receivers 20 m inside all four edges, source 60 m from the top-left corner. The reference
    # pads the physical domain by 1 km on every side, so its own PML reflections travel ≥ 2 km farther.
    c0, h, L = 2000.0, 10.0, 500.0
    xr = collect(20.0:20.0:480.0)
    acq = Acquisition([60.0], [60.0],
                      vcat(fill(20.0, length(xr)), fill(480.0, length(xr)), xr, xr),
                      vcat(xr, xr, fill(20.0, length(xr)), fill(480.0, length(xr))))
    function data(f, npml, pad)
        ω = 2π * f
        n = round(Int, (L + 2pad) / h) + 1
        g = Grid2D(n, n, h, -pad, -pad, (npml, npml, npml, npml))
        op = HelmholtzPML(g, Material(fill(c0, n, n), fill(1000.0, n, n)), ω, CHEN2013_GLOBAL, chen_stretch(g, ω; f0=10.0))
        return predict_data(op, acq, [1.0])
    end
    for f in (5.0, 10.0, 20.0)
        dref = data(f, 40, 1000.0)
        errs = [norm(data(f, np_, 0.0) - dref) / norm(dref) for np_ in (5, 10, 20, 40)]
        @info "PML reflection" f errs
        @test issorted(errs, rev=true)
        @test errs[3] < 1e-3                 # 20 cells: measured ≤ 3.7e-4
        @test errs[4] < 1e-5                 # 40 cells: measured ≤ 3.7e-6
    end
end
