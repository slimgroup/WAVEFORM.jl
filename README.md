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

Notes
-----
- The 2D stencils use the exp(+iωt) time convention (outgoing waves ~ conj(H0(kr))), while the 3D 27-point stencil uses exp(-iωt) (outgoing waves ~ exp(+ikr)/r).
- With `cut_pml=true`, model-space quantities are restricted to the physical domain by discarding the PML. The misfit gradient then ignores the PML-zone sensitivity of the boundary cells, so it is approximate there. Use `cut_pml=false` for an exact gradient (it passes a Taylor test).
- The examples call `JOLI.jo_type_mismatch_error_set(false)`, because some operators intentionally map real model perturbations to complex wavefields.
