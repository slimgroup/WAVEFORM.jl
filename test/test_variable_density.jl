@testset "Variable-density nine-point Helmholtz" begin
    include("variable_density/test_grid.jl")
    include("variable_density/test_pml.jl")
    include("variable_density/test_materials.jl")
    include("variable_density/test_chen9.jl")
end
