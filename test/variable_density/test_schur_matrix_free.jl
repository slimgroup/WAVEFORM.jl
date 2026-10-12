@testset "Schur: matrix-free actions" begin
    (; g, St, A, acq, B, R, q, u, d) = schur_fixture(); rel = schur_rel
    @testset "matrix-free Schur action, adjoint and reconstruction (stride $stride)" for stride in (2, 3, 4)
        s = build_coarse_fine_maps(g, acquisition_support(B, R); stride, retain_support=false)
        sr = schur_reduction(A, B, R, s)
        # matrix-free Schur action / adjoint against the dense oracle
        fb = FineBlockLU(St, s)
        v = randn(ComplexF64, length(s.ic)); w = randn(ComplexF64, length(s.ic))
        Sv = apply_schur!(similar(v), St, s, fb, v)
        Sw = apply_schur_adj!(similar(w), St, s, fb, w)
        @test rel(Sv, sr.S * v) < 1e-10
        @test rel(Sw, sr.S' * w) < 1e-10
        @test abs(dot(w, Sv) - dot(Sw, v)) < 1e-10 * norm(w) * norm(Sv)
        # matrix-free reduced RHS, retained solve and reconstruction
        f = Vector{ComplexF64}(B * q)
        fc = reduced_rhs!(zeros(ComplexF64, length(s.ic)), St, s, fb, f)
        @test rel(fc, sr.Beff * q) < 1e-10
        uc = sr.S \ fc
        @test rel(reconstruct_field(St, s, fb, f, uc), u) < 1e-10
    end
    @testset "solver/split mismatch (review)" begin
        s = build_coarse_fine_maps(g; stride=2)
        fb = FineBlockLU(St, s)
        r = copy(s.retained); r[s.ic[end]], r[s.jf[1]] = false, true          # same N_f, different split
        s2 = CoarseFineSplit(s.grid_size, r)
        v = randn(ComplexF64, length(s2.ic)); f = randn(ComplexF64, length(s2))
        @test_throws ArgumentError apply_schur!(similar(v), St, s2, fb, v)
        @test_throws ArgumentError apply_schur_adj!(similar(v), St, s2, fb, v)
        @test_throws ArgumentError reduced_rhs!(similar(v), St, s2, fb, f)
        @test_throws ArgumentError reconstruct_field(St, s2, fb, f, v)
    end
end
