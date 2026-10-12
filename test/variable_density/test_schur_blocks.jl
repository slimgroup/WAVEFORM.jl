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
        @test isapprox(FineBlockLU(Ae, s4; rcond_min=0.0).rcond, exact_e; rtol=1e-6)   # estimator accuracy
        @test_throws IllConditionedFineBlock FineBlockLU(Ae, s4)            # below the default 1e-8
        # nothing to eliminate
        s1 = build_coarse_fine_maps(g; stride=1)
        @test FineBlockLU(A, s1).rcond == 1
    end
    @testset "lossless fine block at an interior resonance (review)" begin
        # With the physical boundary ring retained, every eliminated row is a physical row away
        # from the PML: A_ff(ω) = K - ω² M is real and singular at the pencil's eigenfrequencies,
        # while A (damped by the PML) stays well conditioned. The default guard must fire there.
        gl = Grid2D(11, 11, 10.0, 0.0, 0.0, (5, 5, 5, 5))
        ml = Material(fill(2000.0, 11, 11), fill(1000.0, 11, 11))
        zr, xr = physical_z_range(gl), physical_x_range(gl); L = LinearIndices(comp_size(gl))
        ring = [L[i, j] for i in zr, j in xr if i in (first(zr), last(zr)) || j in (first(xr), last(xr))]
        sl = build_coarse_fine_maps(gl, ring; stride=4)
        Aω(ω) = assemble_sparse(HelmholtzPML(gl, ml, ω, CHEN2013_GLOBAL, chen_stretch(gl, ω; f0=10.0)))
        ω1, ω2 = 2π * 10.0, 2π * 20.0
        Aff1, Aff2 = Matrix(block_matrix(Aω(ω1), sl, :f, :f)), Matrix(block_matrix(Aω(ω2), sl, :f, :f))
        M = (Aff1 - Aff2) / (ω2^2 - ω1^2); K = Aff1 + ω1^2 * M
        @test iszero(imag(K)) && iszero(imag(M))                           # lossless
        ωs = sqrt.(filter(>(0), real(eigvals(real(K), real(M)))))
        ωr = ωs[argmin(abs.(ωs .- 2π * 20.0))]
        Ar = Aω(ωr)
        @test rcond_estimate(Ar, lu(Ar)) > 1e-4                           # the full operator is fine
        err = try FineBlockLU(Ar, sl); nothing catch e; e end
        @test err isa IllConditionedFineBlock && err.threshold == 1e-8 && err.rcond < 1e-8
        @test FineBlockLU(Aω(1.01 * ωr), sl).rcond > 1e-8                  # 1 % away it is accepted
    end
end
