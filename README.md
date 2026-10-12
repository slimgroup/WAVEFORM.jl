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
The `examples/` directory has its own environment, which adds PythonPlot for plotting and OptimPackNextGen for the SPG optimizer. OptimPackNextGen is not in the General registry and is pulled from GitHub through `[sources]`:
```
cd examples
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. simple_inversion.jl
julia -t auto --project=. -e 'using IJulia; notebook(dir=".")'   # for the notebooks, if IJulia is installed
```
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
Conventions and an audit of the legacy 2D code are documented in the NonLocal repository
(`docs/conventions.md`).

Notes
-----
- The 2D stencils use the exp(+iωt) time convention (outgoing waves ~ conj(H0(kr))), while the 3D 27-point stencil uses exp(-iωt) (outgoing waves ~ exp(+ikr)/r).
- With `cut_pml=true`, model-space quantities are restricted to the physical domain by discarding the PML. The misfit gradient then ignores the PML-zone sensitivity of the boundary cells, so it is approximate there. Use `cut_pml=false` for an exact gradient (it passes a Taylor test).
- The examples call `JOLI.jo_type_mismatch_error_set(false)`, because some operators intentionally map real model perturbations to complex wavefields.
