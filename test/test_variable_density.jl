@testset "Variable-density nine-point Helmholtz" begin
    include("variable_density/test_grid.jl")
    include("variable_density/test_pml.jl")
    include("variable_density/test_materials.jl")
    include("variable_density/test_chen9.jl")
    include("variable_density/test_operator.jl")
    include("variable_density/test_stencil.jl")
    include("variable_density/test_constant_density_reference.jl")
    include("variable_density/test_acquisition.jl")
end
