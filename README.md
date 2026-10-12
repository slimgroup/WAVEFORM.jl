WAVEFORM - a softWAre enVironmEnt For nOnlinear inveRse probleMs
================================================================

[![CI](https://github.com/slimgroup/WAVEFORM.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/slimgroup/WAVEFORM.jl/actions/workflows/CI.yml)

Waveform is a flexible and modular approach to solving PDE-constrained inverse problems that aims to
- accurately reflect the underlying mathematics of the problem
- produce *readable* code that reflects the underlying mathematics
- scales easily from small 2D problems to large-scale 3D problems with *minimal* code modifications
- allow users to integrate their own preconditioners, stencils, linear solvers, etc. easily in to the entire inversion framework

For more details, as well as the full design of the software, see (https://arxiv.org/abs/1703.09268) and for the original Matlab implementation, see (https://github.com/slimgroup/WAVEFORM). All functions are written by Curt Da Silva (curt.dasilva@gmail.com), unless indicated otherwise in the function documentation.

Installation
------------
Requires Julia 1.11 or newer (tested on 1.12 and 1.13). To add it to your environment:
```
julia -e 'using Pkg; Pkg.add(url="https://github.com/slimgroup/WAVEFORM.jl")'
```
or, to develop from a clone of this repository:
```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
JOLI (installed automatically from the General registry) uses PythonCall, which installs a private Python through CondaPkg on first load. `CondaPkg.toml` caps Python below 3.15, which PythonCall cannot load yet.

If your global Julia environment contains packages with conflicting CondaPkg requirements, CondaPkg merges them in and the Python install can fail. In that case run with `JULIA_LOAD_PATH="@:@stdlib"` to keep the global environment out of the load path.

Run the tests with
```
julia -t auto --project=. -e 'using Pkg; Pkg.test()'
```

Examples
--------
The `examples/` directory has its own environment, which adds PythonPlot for plotting, IJulia for the notebooks and OptimPackNextGen for the SPG optimizer. OptimPackNextGen is not in the General registry and is pulled from GitHub through `[sources]`:
```
cd examples
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. simple_inversion.jl
```
The notebooks are stored with their outputs, so the figures show on GitHub. To run them, register a Jupyter kernel that starts in the notebook's directory with this environment (once), then open them with Jupyter:
```
julia --project=. -e 'using IJulia; installkernel("Julia", "--project=@.")'
jupyter lab      # or: julia --project=. -e 'using IJulia; notebook(dir=".")'
```
- `3D Helmholtz Example.ipynb`: point-source wavefield of the 27-point operator, orthogonal slices and a comparison with the analytic Green's function.
- `Simple 2D FWI Example.ipynb`: transmission FWI of a box anomaly; data, misfit and model-error histories, inverted models, errors and profiles, with and without frequency continuation.

PythonPlot installs matplotlib through CondaPkg into `examples/.CondaPkg`. If the repository lives in a cloud-synced folder (Dropbox, iCloud) that does not preserve that environment, point CondaPkg elsewhere, e.g. `JULIA_CONDAPKG_ENV=$HOME/.julia/conda_environments/waveform-examples`; pass it to the kernel with `installkernel(...; env=Dict("JULIA_CONDAPKG_ENV" => ...))`.

Only non-distributed examples are available.

Variable-density 2D Helmholtz
-----------------------------
`src/VariableDensity/` provides the optimized nine-point (Chen et al. 2013) Helmholtz operator with
variable velocity and density and a stretched-coordinate PML, using the exp(-iωt) convention:
```julia
using WAVEFORM
g  = Grid2D(101, 101, 10.0, 0.0, 0.0, (40, 40, 40, 40))         # nz, nx, h, oz, ox, PML widths
m  = Material(fill(2000.0, 101, 101), fill(1000.0, 101, 101))     # velocity (m/s), density (kg/m³)
ω  = 2π * 10.0
op = HelmholtzPML(g, m, ω, select_chen_weights(points_per_wavelength(2000, 2000, 10, 10, 10)...), chen_stretch(g, ω; f0=10.0))
A  = assemble_sparse(op)                # sparse oracle; NinePointStencil(op) gives the matrix-free form
d  = predict_data(op, Acquisition([500.0], [500.0], [800.0], [500.0]))
```
Discretization: `A u = -Q_z D_x u - Q_x D_z u - ω² W(μ u)` with face buoyancy from the harmonic mean
of nodal buoyancies, α_x = b s_z/s_x, α_z = b s_x/s_z and μ = a s_x s_z (stretch s = 1 + iσ/ω, Chen
profile σ = 2π a0 f0 (l/L)², per side). Point sources are mass-consistent, `B = W Pᵀ/h²` with the same
nine-point average `W` (`mass_matrix`), so the source amplitude does not depend on the grid; a source
must be at least one node away from the PML. With ρ = 1 and the legacy PML profile the operator equals
`-conj(helm2d_chen2013)` (and, with `FIVE_POINT` weights, `-conj(helm2d_7pt)`) to round-off.

Audit of the legacy 2D code (`helm2d_chen2013`, `helm2d_7pt`, `pml_func1d`):
- Fixed: Table 1 typos (G ∈ [3,4]: b = 0.7427, not 0.73427; G ∈ [2.5,3]: d = 0.4444, not 0.444); PML
  widths are read from the columns of `npml = [z_lo x_lo; z_hi x_hi]`; G outside [2.5, 400] raises an
  `ArgumentError` instead of an `UndefVarError`; no NaN when a dimension has no PML.
- Unchanged, for backward compatibility: `pml_func1d` divides σ by the frequency in Hz instead of ω
  (2π stronger damping than the paper), its lower-side distance is one cell short, and both sides use
  the larger thickness max(nb); the weight selection G = min(v)/(h f) assumes velocity in m/s.

### Coarse/fine reduction

`build_coarse_fine_maps` splits the unknowns into retained (`c`) and eliminated (`f`) sets: every
`stride`-th physical node (parity anchored at physical node (1,1)), all PML nodes and the
acquisition support. `schur_reduction` builds the exact Schur complement `S = A_cc - A_cf A_ff⁻¹ A_fc`
with `B_eff`, `R_eff` and `D_direct` for small problems; `apply_schur!`/`apply_schur_adj!` apply `S`
and `S*` matrix-free. An ill-conditioned `A_ff` raises `IllConditionedFineBlock` instead of being
regularized.
```julia
s  = build_coarse_fine_maps(g, acquisition_support(B, R); stride=2)
sr = schur_reduction(assemble_sparse(op), B, R, s)
d  = reduced_data(sr, q)               # equals R A⁻¹ B q
```

Notes
-----
- The 2D stencils use the exp(+iωt) time convention (outgoing waves ~ conj(H0(kr))), while the 3D 27-point stencil uses exp(-iωt) (outgoing waves ~ exp(+ikr)/r).
- With `cut_pml=true`, model-space quantities are restricted to the physical domain by discarding the PML. The misfit gradient then ignores the PML-zone sensitivity of the boundary cells, so it is approximate there. Use `cut_pml=false` for an exact gradient (it passes a Taylor test).
- The examples call `JOLI.jo_type_mismatch_error_set(false)`, because some operators intentionally map real model perturbations to complex wavefields.
