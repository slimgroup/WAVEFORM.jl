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
        kh = ω / c0 * h
        @test norm(d - G) / norm(G) < 0.12 * kh^2              # measured 0.083–0.085 (kh)²
        cfit = (d' * G) / (d' * d)                              # one complex scale (source normalization)
        @test norm(cfit * d - G) / norm(G) < 1e-3               # measured ≤ 2.3e-4
        @test abs(cfit - 1) < 0.12 * kh^2
    end
end
