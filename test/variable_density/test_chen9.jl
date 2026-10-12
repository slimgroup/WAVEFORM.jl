@testset "Chen nine-point weights and dispersion" begin
    # paper values, Chen et al. (2013) eq. (3.22) and Table 1
    @test (CHEN2013_GLOBAL.β, CHEN2013_GLOBAL.w1, CHEN2013_GLOBAL.w2) == (0.7926, 0.3768, -0.0064)
    @test CHEN2013_GLOBAL.w0 ≈ 1 - 0.3768 + 0.0064
    table = [(2.5, 3.0, 0.6803, 0.4444, 0.0008), (3.0, 4.0, 0.7427, 0.4088, -0.0036),
             (4.0, 5.0, 0.7840, 0.3832, -0.0060), (5.0, 6.0, 0.8020, 0.3712, -0.0072),
             (6.0, 8.0, 0.8133, 0.3637, -0.0075), (8.0, 10.0, 0.8219, 0.3578, -0.0078),
             (10.0, 400.0, 0.8271, 0.3540, -0.0080)]
    for ((lo, hi, w), (lo2, hi2, b, d, e)) in zip(CHEN2013_REFINED, table)
        @test (lo, hi, w.β, w.w1, w.w2) == (lo2, hi2, b, d, e)
        @test w.w0 + w.w1 + w.w2 ≈ 1
    end
    # selection policy
    @test select_chen_weights(12.0, 50.0) == CHEN2013_REFINED[7][3]
    @test select_chen_weights(3.0, 3.0) == CHEN2013_REFINED[1][3]   # shared endpoint: lower row
    @test select_chen_weights(4.5, 9.0) == CHEN2013_GLOBAL         # straddles rows -> global
    @test_throws ArgumentError select_chen_weights(2.0, 5.0)        # below the table
    @test_throws ArgumentError select_chen_weights(3.0, 4.5)        # straddles, not within [4,400]
    @test_throws ArgumentError select_chen_weights(50.0, 500.0)     # above the table
    @test_throws ArgumentError select_chen_weights(9.0, 8.0)
    @test points_per_wavelength(1500.0, 3000.0, 5.0, 15.0, 10.0) == (10.0, 60.0)
    # dispersion from our symbol reproduces the paper's closed form eq. (3.12)
    paper(b, d, e, G, θ) = (P = cos(2π / G * cos(θ)); Q = cos(2π / G * sin(θ));
        G / (2π) * sqrt((4b + 2(1 - 2b) * (P + Q) + 4(b - 1) * P * Q) / ((1 - d - e) + d / 2 * (P + Q) + e * P * Q)))
    Gs = range(4, 400, length=2000); θs = range(0, π / 2, length=91)
    @test maximum(abs(normalized_phase_velocity(CHEN2013_GLOBAL, G, θ) - paper(0.7926, 0.3768, -0.0064, G, θ)) for G in Gs, θ in θs) < 1e-10
    # dispersion claims (measured: 4.55e-3 global, 9.97e-2 five-point)
    @test maximum(abs(normalized_phase_velocity(CHEN2013_GLOBAL, G, θ) - 1) for G in Gs, θ in θs) < 5e-3
    @test maximum(abs(normalized_phase_velocity(FIVE_POINT, G, θ) - 1) for G in Gs, θ in θs) > 9e-2
    for (lo, hi, w) in CHEN2013_REFINED
        Gi = range(lo, hi, length=200)
        @test maximum(abs(normalized_phase_velocity(w, G, θ) - 1) for G in Gi, θ in θs) <
              maximum(abs(normalized_phase_velocity(CHEN2013_GLOBAL, G, θ) - 1) for G in Gi, θ in θs)
    end
end
