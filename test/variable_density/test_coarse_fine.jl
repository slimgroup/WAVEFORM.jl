@testset "Coarse/fine maps" begin
    g = Grid2D(7, 9, 10.0, 0.0, 0.0, (2, 3, 1, 2))             # comp grid 12 × 12, N = 144
    s = build_coarse_fine_maps(g; stride=2)
    Pc, Pf = restriction(s, :c), restriction(s, :f)
    Ec, Ef = injection(s, :c), injection(s, :f)
    sm = split_summary(s, g)
    # stated dimensions: 4 × 5 retained physical nodes + all 144 - 63 = 81 PML nodes
    @test (sm.N, sm.N_c, sm.N_f) == (144, 101, 43)
    @test sm.pml_retained == 81 && sm.physical_retained_fraction ≈ 20 / 63
    @test size(Pc) == (101, 144) && size(Pf) == (43, 144) && size(Ec) == (144, 101)
    # identity / complement relations (spec Sec. 5.1), exact
    @test Pc * Ec == I(101) && Pf * Ef == I(43)
    @test nnz(Pc * Ef) == 0 && nnz(Pf * Ec) == 0
    @test Ec * Pc + Ef * Pf == I(144)
    @test Ec == transpose(Pc)
    u = randn(ComplexF64, 144)
    @test Ec * (Pc * u) + Ef * (Pf * u) == u
    # parity anchored at physical node (1,1): the corners of the physical domain are retained
    keep = reshape(s.retained, comp_size(g))
    zr, xr = physical_z_range(g), physical_x_range(g)
    @test keep[first(zr), first(xr)] && keep[last(zr), last(xr)] && !keep[first(zr)+1, first(xr)]
    # global-index invariance: the retained physical nodes depend only on physical coordinates
    for npml in ((0, 0, 0, 0), (5, 1, 4, 3)), stride in (2, 3, 4)
        g2 = Grid2D(7, 9, 10.0, 0.0, 0.0, npml)
        k1 = reshape(build_coarse_fine_maps(g; stride).retained, comp_size(g))[zr, xr]
        k2 = reshape(build_coarse_fine_maps(g2; stride).retained, comp_size(g2))[physical_z_range(g2), physical_x_range(g2)]
        @test k1 == k2
    end
    # acquisition support is retained on request, eliminated otherwise
    sup = [LinearIndices(comp_size(g))[first(zr)+1, first(xr)+1]]
    @test build_coarse_fine_maps(g, sup).retained[sup[1]]
    @test !build_coarse_fine_maps(g, sup; retain_support=false).retained[sup[1]]
    # PML elimination and the trivial splits
    @test split_summary(build_coarse_fine_maps(g; retain_pml=false), g).pml_retained == 0
    @test split_summary(build_coarse_fine_maps(g; stride=1), g).N_f == 0
    @test_throws ArgumentError build_coarse_fine_maps(g; stride=0)
    @test_throws ArgumentError build_coarse_fine_maps(g, [145])
    @test_throws ArgumentError restriction(s, :x)
    @test_throws DimensionMismatch CoarseFineSplit((3, 3), trues(8))
end
