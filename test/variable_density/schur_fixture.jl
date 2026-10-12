# Shared small problem for the Schur tests: variable velocity, a density jump, unequal PML,
# non-square grid with non-zero origin, two sources and five receivers off the grid nodes.
function schur_fixture()
    Random.seed!(21)
    # small variable-density problem, unequal PML, non-square, non-zero origin
    g = Grid2D(15, 13, 10.0, 100.0, -60.0, (4, 5, 3, 6))
    zp = [comp_z(g, i) for i in physical_z_range(g)]; xp = [comp_x(g, j) for j in physical_x_range(g)]
    c = [2000 + 300 * sin(z / 40) * cos(x / 55) for z in zp, x in xp]
    ρ = [x > 0 ? 2200.0 : 1500.0 for z in zp, x in xp]                    # density jump
    ω = 2π * 8.0
    op = HelmholtzPML(g, Material(c, ρ), ω, CHEN2013_GLOBAL, chen_stretch(g, ω; f0=8.0))
    St = NinePointStencil(op); A = assemble_sparse(St)
    acq = Acquisition([135.0, 175.0], [-25.0, 15.0], [115.0, 155.0, 195.0, 215.0, 125.0], [-45.0, -5.0, 35.0, 55.0, 25.0])
    B = source_matrix(g, acq, CHEN2013_GLOBAL); R = receiver_matrix(g, acq)
    q = randn(ComplexF64, 2)
    u = lu(A) \ Vector{ComplexF64}(B * q); d = R * u
    return (; g, St, A, acq, B, R, q, u, d)
end
schur_rel(a, b) = norm(a - b) / norm(b)
