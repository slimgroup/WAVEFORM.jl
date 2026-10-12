@testset "Grid2D" begin
    g = Grid2D(7, 5, 10.0, 100.0, -50.0, (2, 3, 1, 4))
    @test comp_size(g) == (12, 10)
    @test ncomp(g) == 120
    @test physical_z_range(g) == 3:9
    @test physical_x_range(g) == 2:6
    @test comp_z(g, 3) == 100.0          # first physical node
    @test comp_x(g, 2) == -50.0
    @test comp_z(g, 1) == 80.0           # PML nodes continue the spacing
    @test comp_x(g, 10) == -50.0 + 8 * 10.0
    @test z_index(g, 100.0) == 3.0
    @test x_index(g, -45.0) == 2.5
    @test z_index(g, comp_z(g, 7.25)) ≈ 7.25
    @test_throws ArgumentError Grid2D(2, 5, 10.0, 0.0, 0.0, (0, 0, 0, 0))
    @test_throws ArgumentError Grid2D(5, 5, 0.0, 0.0, 0.0, (0, 0, 0, 0))
    @test_throws ArgumentError Grid2D(5, 5, 1.0, 0.0, 0.0, (0, -1, 0, 0))
end
