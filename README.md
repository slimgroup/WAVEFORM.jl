# WAVEFORM - a softWAre enVironmEnt For nOnlinear inveRse probleMs

Waveform is a flexible and modular approach to solving PDE-constrained inverse problems that aims to
- accurately reflect the underlying mathematics of the problem
- produce *readable* code that reflects the underlying mathematics
- scales easily from small 2D problems to large-scale 3D problems with *minimal* code modifications
- allow users to integrate their own preconditioners, stencils, linear solvers, etc. easily in to the entire inversion framework

For more details, as well as the full design of the software, see (https://arxiv.org/abs/1703.09268) and for the original Matlab implementation, see (https://github.com/slimgroup/WAVEFORM). All functions are written by Curt Da Silva (curt.dasilva@gmail.com), unless indicated otherwise in the function documentation.

## Installation
In the Julia terminal, type
```
Pkg.clone("git@github.com:slimgroup/JOLI.jl.git")
Pkg.clone("git@github.com:slimgroup/WAVEFORM.jl.git")
```

## Examples
Examples can be run from the examples/ directory. Currently there are only non-distributed examples available, but that will change in future updates.

