export HelmholtzPML, reference_apply

"""
    HelmholtzPML(grid, material, ω, weights, stretch)

Definition of the variable-density nine-point PML action (spec Sec. 4)

    A_h u = -Q_z D_x u - Q_x D_z u - ω² W(μ u),

with α_x = b s_z/s_x on x-faces, α_z = b s_x/s_z on z-faces and μ = a s_x s_z on nodes,
b = 1/ρ, a = 1/(ρ c²). Face buoyancy is the harmonic mean of the two adjacent nodal
buoyancies (outer ghost faces use the boundary node). Coefficient values move with the
transversely shifted row; W averages the product μu.
"""
struct HelmholtzPML
    grid::Grid2D
    ω::Float64
    weights::Chen9Weights
    αx::Matrix{ComplexF64}   # (nzc, nxc+1): αx[iz, k] on the x-face at ξx = k - 1/2
    αz::Matrix{ComplexF64}   # (nzc+1, nxc): αz[k, ix] on the z-face at ξz = k - 1/2
    μ::Matrix{ComplexF64}    # (nzc, nxc)
end

_harmonic(p, q) = 2p * q / (p + q)

function HelmholtzPML(g::Grid2D, m::Material, ω::Real, w::Chen9Weights, st::Stretch2D)
    size(m.c) == (g.nz, g.nx) || throw(DimensionMismatch("material is $(size(m.c)), grid is ($(g.nz), $(g.nx))"))
    ω > 0 || throw(ArgumentError("ω must be positive, got $ω"))
    ρ = extend_to_pml(g, m.rho)
    c = extend_to_pml(g, m.c)
    b = 1 ./ ρ
    a = 1 ./ (ρ .* c .^ 2)
    nzc, nxc = comp_size(g)
    szn = [st.sz(iz) for iz in 1:nzc]
    sxn = [st.sx(ix) for ix in 1:nxc]
    szf = [st.sz(k - 0.5) for k in 1:nzc+1]
    sxf = [st.sx(k - 0.5) for k in 1:nxc+1]
    bx = hcat(b[:, 1], b, b[:, end])
    bz = vcat(b[1:1, :], b, b[end:end, :])
    bfx = _harmonic.(bx[:, 1:end-1], bx[:, 2:end])
    bfz = _harmonic.(bz[1:end-1, :], bz[2:end, :])
    αx = bfx .* szn ./ transpose(sxf)
    αz = bfz .* transpose(sxn) ./ szf
    μ = a .* szn .* transpose(sxn)
    return HelmholtzPML(g, Float64(ω), w, αx, αz, μ)
end

"`shifted(v, dz, dx)[iz, ix] = v[iz+dz, ix+dx]`, zero outside the grid."
function shifted(v::AbstractMatrix, dz::Int, dx::Int)
    out = zero(v)
    nz, nx = size(v)
    for ix in max(1, 1 - dx):min(nx, nx - dx), iz in max(1, 1 - dz):min(nz, nz - dz)
        out[iz, ix] = v[iz+dz, ix+dx]
    end
    return out
end

"""
    reference_apply(op, u)

Apply `A_h` directly from its definition (spec Sec. 4). Slow but transparent: this is the
oracle for the stencil, sparse, and later Devito implementations. Values of `u`, `D_x u`,
`D_z u` and `μu` outside the computational grid are zero (Dirichlet closure).
"""
function reference_apply(op::HelmholtzPML, u::AbstractMatrix)
    nzc, nxc = size(op.μ)
    size(u) == (nzc, nxc) || throw(DimensionMismatch("u is $(size(u)), grid is ($nzc, $nxc)"))
    h2 = op.grid.h^2
    w = op.weights
    uE, uW = shifted(u, 0, 1), shifted(u, 0, -1)
    uS, uN = shifted(u, 1, 0), shifted(u, -1, 0)
    Dx = (op.αx[:, 2:end] .* (uE .- u) .- op.αx[:, 1:end-1] .* (u .- uW)) ./ h2
    Dz = (op.αz[2:end, :] .* (uS .- u) .- op.αz[1:end-1, :] .* (u .- uN)) ./ h2
    QzDx = w.β .* Dx .+ (1 - w.β) / 2 .* (shifted(Dx, -1, 0) .+ shifted(Dx, 1, 0))
    QxDz = w.β .* Dz .+ (1 - w.β) / 2 .* (shifted(Dz, 0, -1) .+ shifted(Dz, 0, 1))
    mu = op.μ .* u
    Wmu = w.w0 .* mu .+
          w.w1 / 4 .* (shifted(mu, -1, 0) .+ shifted(mu, 1, 0) .+ shifted(mu, 0, -1) .+ shifted(mu, 0, 1)) .+
          w.w2 / 4 .* (shifted(mu, -1, -1) .+ shifted(mu, -1, 1) .+ shifted(mu, 1, -1) .+ shifted(mu, 1, 1))
    return .-QzDx .- QxDz .- op.ω^2 .* Wmu
end
