@testset "Schur: blocks and fine-block conditioning" begin
    (; g, St, A, acq, B, R, q, u, d) = schur_fixture(); rel = schur_rel
    @testset "block definitions (stride $stride)" for stride in (2, 3)
        s = build_coarse_fine_maps(g; stride)
        for α in (:c, :f), β in (:c, :f)
            Aαβ = block_matrix(A, s, α, β)
            @test Aαβ == restriction(s, α) * A * injection(s, β)
            v = randn(ComplexF64, size(Aαβ, 2))
            @test rel(apply_block!(zeros(ComplexF64, size(Aαβ, 1)), St, s, α, β, v), Aαβ * v) < 1e-13
        end
    end
    @testset "fine-block conditioning" begin
        s = build_coarse_fine_maps(g; stride=2)
        Aff = Matrix(block_matrix(A, s, :f, :f))
        exact = 1 / cond(Aff, 1)
        est = FineBlockLU(A, s).rcond
        @test exact * (1 - 1e-10) <= est <= 3 * exact                     # estimator: lower bound on ‖A_ff⁻¹‖₁
        # A invertible, A_ff singular: detected and reported, never regularized
        As = sparse(ComplexF64[2 1 0; 1 0 1; 0 1 2]); s3 = CoarseFineSplit((3, 1), [true, false, true])
        @test abs(det(Matrix(As))) > 1
        err = try FineBlockLU(As, s3); nothing catch e; e end
        @test err isa IllConditionedFineBlock && err.rcond == 0
        # nearly singular 2×2 A_ff (conditioning is scale-invariant, so a 1×1 block never is)
        Ae = sparse(ComplexF64[3 1 0 0; 1 1 1 0; 0 1 1+1e-9 1; 0 0 1 3]); s4 = CoarseFineSplit((4, 1), [true, false, false, true])
        exact_e = 1 / cond(Matrix(Ae)[2:3, 2:3], 1)                        # ≈ 2.5e-10
        @test abs(det(Matrix(Ae))) > 1                                     # the full matrix is fine
        @test_throws IllConditionedFineBlock FineBlockLU(Ae, s4; rcond_min=1e-6)
        @test isapprox(FineBlockLU(Ae, s4).rcond, exact_e; rtol=1e-6)     # passes the default 1e-12
        # nothing to eliminate
        s1 = build_coarse_fine_maps(g; stride=1)
        @test FineBlockLU(A, s1).rcond == 1
    end
end
