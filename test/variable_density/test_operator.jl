@testset "Operator definition" begin
    Random.seed!(11)
    ω = 2π * 8.0
    # plane-wave symbol (homogeneous, no PML): exact on interior nodes
    g = Grid2D(21, 17, 5.0, 0.0, 0.0, (0, 0, 0, 0))
    c0, ρ0 = 1800.0, 1500.0
    op = HelmholtzPML(g, Material(fill(c0, 21, 17), fill(ρ0, 21, 17)), ω, CHEN2013_GLOBAL, no_stretch())
    ξ, η = 0.7, -0.4                                      # kx h, kz h
    pw = [cis(ξ * ix + η * iz) for iz in 1:21, ix in 1:17]
    λ = nine_point_symbol(CHEN2013_GLOBAL, 1 / ρ0, 1 / (ρ0 * c0^2), ω, 5.0, ξ, η)
    Apw = reference_apply(op, pw)
    @test maximum(abs.(Apw[2:end-1, 2:end-1] .- λ .* pw[2:end-1, 2:end-1])) < 1e-12 * abs(λ)
    # density scaling: ρ -> κρ at fixed c scales b and a by 1/κ, hence A by 1/κ (with PML)
    g2 = Grid2D(9, 11, 10.0, 0.0, 0.0, (3, 2, 4, 1))
    c = 2000 .+ 300 .* rand(9, 11); ρ = 1000 .+ 900 .* rand(9, 11)
    st = chen_stretch(g2, ω; f0=10.0)
    u = randn(ComplexF64, comp_size(g2))
    A1 = reference_apply(HelmholtzPML(g2, Material(c, ρ), ω, CHEN2013_GLOBAL, st), u)
    A3 = reference_apply(HelmholtzPML(g2, Material(c, 3 .* ρ), ω, CHEN2013_GLOBAL, st), u)
    @test norm(A3 .- A1 ./ 3) < 1e-12 * norm(A1)
    # face buoyancy is the harmonic mean of adjacent nodal buoyancies (spec: positive face rule)
    opv = HelmholtzPML(g2, Material(c, ρ), ω, CHEN2013_GLOBAL, no_stretch())
    b = 1 ./ extend_to_pml(g2, ρ)
    iz, k = 5, 6                                           # x-face between comp nodes 5 and 6
    @test opv.αx[iz, k] ≈ 2b[iz, k-1] * b[iz, k] / (b[iz, k-1] + b[iz, k])
    @test opv.αz[4, 7] ≈ 2b[3, 7] * b[4, 7] / (b[3, 7] + b[4, 7])
    @test opv.αx[iz, 1] ≈ b[iz, 1]                         # outer ghost face uses the boundary node
    # with PML: α_x = b s_z(row)/s_x(face), μ = a s_x s_z on nodes
    ops = HelmholtzPML(g2, Material(c, ρ), ω, CHEN2013_GLOBAL, st)
    @test ops.αx[2, 3] ≈ opv.αx[2, 3] * st.sz(2) / st.sx(2.5)
    @test ops.μ[1, 1] ≈ opv.μ[1, 1] * st.sz(1) * st.sx(1)
    @test_throws DimensionMismatch reference_apply(op, zeros(ComplexF64, 3, 3))
    @test_throws ArgumentError HelmholtzPML(g, Material(fill(c0, 21, 17), fill(ρ0, 21, 17)), -ω, CHEN2013_GLOBAL, no_stretch())
end
