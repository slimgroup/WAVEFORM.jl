@testset "Constant-density reference: helm2d_chen2013 / helm2d_7pt" begin
    Random.seed!(13)
    h, f0, a0 = 10.0, 10.0, 1.79
    # Build our operator on the padded grid, with WAVEFORM's own (conjugated) PML profile.
    # npml uses helm.jl's layout [z_lo x_lo; z_hi x_hi].
    function ours(v, f, npml, w)
        nzc, nxc = size(v)
        ez = WAVEFORM.pml_func1d(nzc, npml[:, 1], a0, f0, f)
        ex = WAVEFORM.pml_func1d(nxc, npml[:, 2], a0, f0, f)
        st = Stretch2D(ξ -> conj(ez(ξ)), ξ -> conj(ex(ξ)))      # exp(+iωt) -> exp(-iωt)
        gp = Grid2D(nzc, nxc, h, 0.0, 0.0, (0, 0, 0, 0))
        return assemble_sparse(HelmholtzPML(gp, Material(v, ones(nzc, nxc)), 2π * f, w, st))
    end
    chen(v, f, npml) = WAVEFORM.helm2d_chen2013(collect(size(v)), [h, h], npml, f, v, f0, "m/s")[1]
    rel(A, H) = norm(A + conj.(H)) / norm(A)
    nzc, nxc = 47, 43
    npml_eq = 8 * ones(Int, 2, 2)
    # (a) exact match, G ≈ 33: Table 1 row [10,400], where WAVEFORM already agreed with the paper
    v = 2000 .+ 500 .* rand(nzc, nxc); f = 6.0
    G = minimum(v) / (h * f)                                   # WAVEFORM's rule: G = vmin/(h f)
    @test rel(ours(v, f, npml_eq, select_chen_weights(G, G)), chen(v, f, npml_eq)) < 1e-13
    # (b) audit fix 3: G ≈ 3.5 must use the paper's b = 0.7427 (WAVEFORM had 0.73427)
    v34 = 350.0 .+ rand(nzc, nxc); f34 = 10.0
    @test rel(ours(v34, f34, npml_eq, CHEN2013_REFINED[2][3]), chen(v34, f34, npml_eq)) < 1e-13
    # (c) audit fix 4: unequal widths, layout [z_lo x_lo; z_hi x_hi]
    npml_u = [2 5; 3 4]
    @test rel(ours(v, f, npml_u, select_chen_weights(G, G)), chen(v, f, npml_u)) < 1e-13
    @test rel(ours(v, f, npml_u, FIVE_POINT),
              WAVEFORM.helm2d_7pt(collect(size(v)), [h, h], npml_u, f, v, f0, "m/s")[1]) < 1e-13
    # (d) audit fix 5: G outside [2.5, 400] is an explicit error
    @test_throws ArgumentError chen(fill(20.0, nzc, nxc), 10.0, npml_eq)    # G = 0.2
    # (e) audit findings 1-2 are documented, not changed: WAVEFORM's Im s at the outermost lower
    # node is 2π (nb-1)²/nb² times the paper's; at the outermost upper node exactly 2π times
    nb = 8
    ez = WAVEFORM.pml_func1d(nzc, [nb, nb], a0, f0, f)
    g = Grid2D(nzc - 2nb, nxc - 2nb, h, 0.0, 0.0, (nb, nb, nb, nb))
    st = chen_stretch(g, 2π * f; f0=f0, a0=a0)
    @test imag(conj(ez(1.0))) / imag(st.sz(1.0)) ≈ 2π * (nb - 1)^2 / nb^2
    @test imag(conj(ez(Float64(nzc)))) / imag(st.sz(Float64(nzc))) ≈ 2π
    # Review: a dimension without PML (zero width on both sides) must give a finite legacy operator
    npml_z = [5 0; 5 0]
    @test all(isfinite, nonzeros(chen(v, f, npml_z)))
    @test all(isfinite, nonzeros(WAVEFORM.helm2d_7pt(collect(size(v)), [h, h], npml_z, f, v, f0, "m/s")[1]))
    @test rel(ours(v, f, npml_z, select_chen_weights(G, G)), chen(v, f, npml_z)) < 1e-13
end
