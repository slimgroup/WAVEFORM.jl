@testset "Acquisition and forward oracle" begin
    g = Grid2D(11, 9, 10.0, 200.0, -40.0, (3, 2, 4, 1))        # non-square, non-zero origin
    zc = [comp_z(g, i) for i in 1:comp_size(g)[1]]; xc = [comp_x(g, j) for j in 1:comp_size(g)[2]]
    lin = vec([z + 1000x for z in zc, x in xc])
    pts_z = [200.0, 213.0, 295.5, 300.0]; pts_x = [-40.0, -27.5, 0.0, 40.0]
    P = interpolation_matrix(g, pts_z, pts_x)
    @test P * lin ≈ pts_z .+ 1000 .* pts_x                     # Review Focus 5: no z/x swap
    @test all(sum(P, dims=2) .≈ 1)
    # Review Focus 1: last physical node with round-off is accepted (snapped), PML is rejected
    zlast = comp_z(g, last(physical_z_range(g)))
    @test nnz(interpolation_matrix(g, [zlast + 1e-11], [5.0])) == 2   # z snapped to one node, x between two
    @test_throws ArgumentError interpolation_matrix(g, [zlast + 3.0], [0.0])
    @test_throws ArgumentError interpolation_matrix(g, [200.0], [-45.0])
    acq = Acquisition([250.0, 260.0], [0.0, 5.0], [210.0, 290.0, 300.0], [-30.0, 10.0, 40.0])
    w = CHEN2013_GLOBAL
    B = source_matrix(g, acq, w); R = receiver_matrix(g, acq)
    @test size(B) == (ncomp(g), 2) && size(R) == (3, ncomp(g))
    @test vec(sum(B, dims=1)) .* g.h^2 ≈ [1.0, 1.0]            # unit total source strength
    # mass-consistent source: the same W as the operator's mass term, applied to the discrete delta
    P = interpolation_matrix(g, acq.src_z, acq.src_x)
    Wm = mass_matrix(g, w)
    @test B ≈ Wm * transpose(P) ./ g.h^2
    @test nnz(source_matrix(g, Acquisition([250.0], [0.0], Float64[], Float64[]), w)) == 9
    @test source_matrix(g, acq, FIVE_POINT) ≈ transpose(P) ./ g.h^2     # W = I for five-point
    @test_throws ArgumentError source_matrix(g, Acquisition([zlast], [0.0], Float64[], Float64[]), w)  # spread into PML
    # W matches the operator's mass term: A(ω) - A(ω') = -(ω² - ω'²) W diag(μ) (μ = a without PML)
    m0 = Material(fill(2000.0, 11, 9), fill(1000.0, 11, 9))
    A1 = assemble_sparse(HelmholtzPML(g, m0, 2π * 5.0, w, no_stretch()))
    A2 = assemble_sparse(HelmholtzPML(g, m0, 2π * 7.0, w, no_stretch()))
    a0 = 1 / (1000.0 * 2000.0^2)
    @test (A1 - A2) ≈ -((2π * 5.0)^2 - (2π * 7.0)^2) * a0 * Wm
    @test Wm ≈ transpose(Wm)
    sup = acquisition_support(B, R)
    @test issorted(sup) && allunique(sup)
    @test Set(sup) == Set(vcat(findnz(B)[1], findnz(R)[2]))
    @test_throws DimensionMismatch Acquisition([1.0], [1.0, 2.0], [1.0], [1.0])
    # forward oracle: predict_data == R A^{-1} B
    m = Material(fill(2000.0, 11, 9), fill(1000.0, 11, 9))
    op = HelmholtzPML(g, m, 2π * 5.0, CHEN2013_GLOBAL, chen_stretch(g, 2π * 5.0; f0=10.0))
    A = assemble_sparse(op)
    D = predict_data(op, acq)
    @test size(D) == (3, 2)
    @test D ≈ R * (Matrix(A) \ Matrix(B))
    F = factorize_A(op); x = zeros(ComplexF64, ncomp(g)); rhs = Vector{ComplexF64}(B[:, 1])
    solve_A!(x, F, rhs)
    @test norm(A * x - rhs) < 1e-10 * norm(rhs)
    @test predict_data(op, acq, [1.0, 2.0]) ≈ D * [1.0, 2.0]
    # Review: round-off at field-scale coordinates (e.g. UTM northing) is still on-node
    gu = Grid2D(11, 11, 1.0, 6.0e6, 5.0e5, (3, 3, 3, 3))
    zl = comp_z(gu, last(physical_z_range(gu)))
    @test nnz(interpolation_matrix(gu, [nextfloat(zl, 4)], [5.0e5 + 2.5])) == 2
    acqu = Acquisition([6.0e6 + 5], [5.0e5 + 5], [nextfloat(zl, 4)], [5.0e5 + 2.5])
    @test size(receiver_matrix(gu, acqu)) == (1, ncomp(gu))
    acq2 = Acquisition([6.0e6 + 5], [5.0e5 + 5], [zl + 0.2], [5.0e5 + 2.5])   # 0.2 cells into the PML
    @test_throws ArgumentError receiver_matrix(gu, acq2)
    @test nnz(receiver_matrix(gu, acq2; snap=0.3)) == 2                  # snap is forwarded
    op2 = HelmholtzPML(gu, Material(fill(2000.0, 11, 11), fill(1000.0, 11, 11)), 2π * 5.0, CHEN2013_GLOBAL, no_stretch())
    @test length(predict_data(op2, acq2, [1.0]; snap=0.3)) == 1
    @test_throws ArgumentError interpolation_matrix(gu, [zl + 0.1], [5.0e5])
end
