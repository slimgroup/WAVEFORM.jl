@testset "PML stretch" begin
    ω, f0, a0 = 2π * 7.0, 10.0, 1.79
    g = Grid2D(20, 15, 5.0, 0.0, 0.0, (4, 8, 0, 6))       # unequal and zero widths
    st = chen_stretch(g, ω; f0=f0)
    σmax = 2π * a0 * f0
    zr, xr = physical_z_range(g), physical_x_range(g)
    # identity in the physical domain, including the interface nodes
    @test all(st.sz(ξ) == 1 for ξ in first(zr):0.5:last(zr))
    @test all(st.sx(ξ) == 1 for ξ in first(xr):0.5:last(xr))
    # quadratic law per side, with that side's own thickness: s = 1 + iσmax (l/L)^2/ω
    @test st.sz(first(zr) - 2) ≈ 1 + im * σmax * (2 / 4)^2 / ω
    @test st.sz(1.0) ≈ 1 + im * σmax / ω                    # outermost lower-z node: l = L
    @test st.sz(last(zr) + 3) ≈ 1 + im * σmax * (3 / 8)^2 / ω
    @test st.sz(last(zr) + 8.5) ≈ 1 + im * σmax * (8.5 / 8)^2 / ω   # outer ghost face
    # zero-width lower-x side: no stretch even beyond the physical domain
    @test st.sx(0.5) == 1
    @test st.sx(last(xr) + 6) ≈ 1 + im * σmax / ω
    # damping sign for exp(-iωt): Im s ≥ 0, so exp(ik ∫ s dx) decays for k > 0
    @test all(imag(st.sz(ξ)) >= 0 for ξ in 0.5:0.5:comp_size(g)[1]+0.5)
    @test abs(exp(im * 1.0 * sum(st.sx(ξ) for ξ in last(xr):0.5:last(xr)+6) * 0.5)) < 1
    @test no_stretch().sz(3.7) == 1 && no_stretch().sx(-2.0) == 1
    @test_throws ArgumentError chen_stretch(g, -1.0; f0=f0)
    @test_throws ArgumentError chen_stretch(g, ω; f0=0.0)
end
