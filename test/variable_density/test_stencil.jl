@testset "Nine-point stencil" begin
    Random.seed!(12)
    ω = 2π * 8.0
    function check(g, m, st, w)
        op = HelmholtzPML(g, m, ω, w, st)
        S = NinePointStencil(op)
        A = assemble_sparse(S)
        u = randn(ComplexF64, comp_size(g)); v = randn(ComplexF64, comp_size(g))
        ref = reference_apply(op, u)
        y = similar(u); apply_A!(y, S, u)
        ya = similar(v); apply_Aadj!(ya, S, v)
        @test norm(y - ref) < 1e-12 * norm(ref)
        @test norm(A * vec(u) - vec(ref)) < 1e-12 * norm(ref)
        @test abs(dot(v, y) - dot(ya, u)) < 1e-12 * abs(dot(v, y))      # complex dot test
        @test norm(A' * vec(v) - vec(ya)) < 1e-12 * norm(ya)
        @test nnz(A) <= 9 * ncomp(g)
        yv = zeros(ComplexF64, ncomp(g)); apply_A!(yv, S, vec(u))         # vector interface
        @test yv ≈ vec(ref)
        return S
    end
    # heterogeneous, unequal PML widths, non-square grid
    g = Grid2D(13, 11, 10.0, 0.0, 0.0, (3, 4, 2, 5))
    m = Material(2000 .+ 300 .* rand(13, 11), 1000 .+ 800 .* rand(13, 11))
    S = check(g, m, chen_stretch(g, ω; f0=10.0), CHEN2013_GLOBAL)
    # Dirichlet closure: no coefficients reach outside the grid
    @test all(all(iszero, S.coeffs[1, :, k]) for (k, (dz, dx)) in enumerate(NINE_POINT_OFFSETS) if dz == -1)
    @test all(all(iszero, S.coeffs[:, end, k]) for (k, (dz, dx)) in enumerate(NINE_POINT_OFFSETS) if dx == 1)
    # Review Focus 4: extreme density contrast and the smallest grid (below the 3x3 probe period)
    g3 = Grid2D(3, 3, 1.0, 0.0, 0.0, (0, 0, 0, 0))
    ρx = [1.0 1e4 1.0; 1e4 1.0 1e4; 1.0 1e4 1.0]
    S3 = check(g3, Material(fill(1500.0, 3, 3), ρx), no_stretch(), CHEN2013_REFINED[7][3])
    @test all(isfinite, S3.coeffs)
    # five-point ablation has no corner coefficients
    S5 = check(g, m, no_stretch(), FIVE_POINT)
    @test all(iszero, S5.coeffs[:, :, findfirst(==((-1, -1)), NINE_POINT_OFFSETS)])
    # spec-named builder
    @test build_coefficients(m, ω, g, no_stretch(), FIVE_POINT).coeffs == S5.coeffs
end
