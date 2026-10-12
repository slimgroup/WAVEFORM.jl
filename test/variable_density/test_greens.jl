@testset "Point source vs analytic Green's function" begin
    # For constant c, ρ:  -b(Δ + k²)u = δ  =>  u = ρ (i/4) H0^(1)(k r)  (exp(-iωt), outgoing)
    h, n, np_, c0, ρ0 = 10.0, 101, 40, 2000.0, 1800.0
    g = Grid2D(n, n, h, 0.0, 0.0, (np_, np_, np_, np_))
    xr = collect(100.0:10.0:900.0)
    acq = Acquisition([500.0], [500.0], fill(800.0, length(xr)), xr)
    r = hypot.(xr .- 500, 300.0)
    for f in (5.0, 10.0, 15.0)
        ω = 2π * f
        w = select_chen_weights(points_per_wavelength(c0, c0, f, f, h)...)
        op = HelmholtzPML(g, Material(fill(c0, n, n), fill(ρ0, n, n)), ω, w, chen_stretch(g, ω; f0=10.0))
        d = predict_data(op, acq, [1.0])
        G = ρ0 .* (im / 4) .* hankelh1.(0, (ω / c0) .* r)
        # Mass-consistent source W δ/h²: no grid-dependent amplitude factor. With a plain δ/h² the
        # response was scaled by 1/M(kh) ≈ 1 + 0.085 (kh)² (M = symbol of W), which differs per grid.
        @test norm(d - G) / norm(G) < 1e-3                      # measured 4.5e-5, 2.1e-4, 2.4e-4
        cfit = (d' * G) / (d' * d)
        @test abs(cfit - 1) < 1e-3
    end
end
