@testset "Materials" begin
    g = Grid2D(4, 3, 10.0, 0.0, 0.0, (2, 1, 0, 2))
    f = reshape(collect(1.0:12.0), 4, 3)
    F = extend_to_pml(g, f)
    @test size(F) == comp_size(g)
    @test F[physical_z_range(g), physical_x_range(g)] == f
    @test F[1, 1] == f[1, 1] && F[2, 1] == f[1, 1]          # copied upward from the top row
    @test F[end, end] == f[end, end]                        # corner copies the corner node
    @test F[3, end] == f[1, end]                            # right PML copies the last column
    E = extension_matrix(g)
    @test size(E) == (ncomp(g), 12)
    @test E * vec(f) == vec(F)
    # the adjoint sums PML copies onto border nodes: it is not a crop
    counts = reshape(E' * ones(ncomp(g)), 4, 3)
    @test counts[2, 2] == 1                                  # interior node of a 4x3 grid
    @test counts[1, 1] == 3                                  # top-left: itself + 2 above
    @test counts[4, 3] == (1 + 1) * (1 + 2)                  # bottom-right corner block
    @test sum(counts) == ncomp(g)
    m = Material(fill(2000.0, 4, 3), fill(1000.0, 4, 3))
    @test m.c isa Matrix{Float64}
    @test_throws DimensionMismatch Material(ones(4, 3), ones(3, 4))
    @test_throws ArgumentError Material(fill(-1.0, 4, 3), ones(4, 3))
    @test_throws ArgumentError Material(ones(4, 3), fill(NaN, 4, 3))
    @test_throws DimensionMismatch extend_to_pml(g, ones(3, 3))
end
