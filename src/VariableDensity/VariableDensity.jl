# Variable-density optimized nine-point Helmholtz-PML operator in 2D (exp(-iωt) convention).
# Conventions and the audit of the legacy 2D code: README.md, "Variable-density 2D Helmholtz".
include("grid.jl")
include("pml.jl")
include("materials.jl")
include("chen9_weights.jl")
include("operator.jl")
include("stencil.jl")
include("acquisition.jl")
include("forward.jl")
include("coarse_fine.jl")
include("fine_block.jl")
include("schur_oracle.jl")
include("schur_matrix_free.jl")
