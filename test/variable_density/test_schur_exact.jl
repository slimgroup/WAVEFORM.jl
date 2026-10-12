@testset "Schur: exact elimination oracle" begin
    (; g, St, A, acq, B, R, q, u, d) = schur_fixture(); rel = schur_rel
    @testset "exact elimination, eliminated acquisition support (stride $stride)" for stride in (2, 3, 4)
        s = build_coarse_fine_maps(g, acquisition_support(B, R); stride, retain_support=false)
        Bf, Rf = B[s.jf, :], R[:, s.jf]
        @test norm(Bf) > 0 && norm(Rf) > 0                                 # general case of Sec. 5.3
        sr = schur_reduction(A, B, R, s)
        @test rel(reduced_data(sr, q), d) < 1e-10                         # data, incl. D_direct
        @test norm(sr.Ddirect) > 0
        ur = reconstruct_field(sr, q)
        @test rel(ur, u) < 1e-10
        @test norm(A * ur - B * q) < 1e-10 * norm(B * q)                  # true full residual
        Q = randn(ComplexF64, 2, 3)                                        # several source batches
        @test rel(reduced_data(sr, Q), R * (lu(A) \ Matrix{ComplexF64}(B * Q))) < 1e-10
    end
    @testset "retained acquisition support" begin
        s = build_coarse_fine_maps(g, acquisition_support(B, R))
        sr = schur_reduction(A, B, R, s)
        @test nnz(B[s.jf, :]) == 0 && nnz(R[:, s.jf]) == 0
        @test iszero(sr.Ddirect) && sr.Reff == Matrix{ComplexF64}(R[:, s.ic])
        @test rel(reduced_data(sr, q), d) < 1e-10
    end
    @testset "trivial and singular splits" begin
        s1 = build_coarse_fine_maps(g; stride=1)                           # nothing eliminated
        @test rel(reduced_data(schur_reduction(A, B, R, s1), q), d) < 1e-10
        As = sparse(ComplexF64[2 1 0; 1 0 1; 0 1 2]); s3 = CoarseFineSplit((3, 1), [true, false, true])
        @test_throws IllConditionedFineBlock schur_reduction(As, ones(3, 1), ones(1, 3), s3)
    end
    @testset "input sizes (review)" begin
        s = build_coarse_fine_maps(g; stride=2)
        @test_throws DimensionMismatch schur_reduction(A, vcat(B, B), R, s)
        @test_throws DimensionMismatch schur_reduction(A, B, hcat(R, R), s)
    end
end
