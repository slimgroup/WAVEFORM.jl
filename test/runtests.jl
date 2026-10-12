using WAVEFORM
using JOLI
using LinearAlgebra
using SparseArrays
using Random
using SpecialFunctions
using ForwardDiff
using Test

JOLI.jo_type_mismatch_error_set(false)
Random.seed!(1234)

# Least-squares slope of log10(err) against log10(h)
convergence_rate(h,err) = (x = log10.(h); y = log10.(err); sum((x.-sum(x)/length(x)).*(y.-sum(y)/length(y)))/sum((x.-sum(x)/length(x)).^2))

@testset "WAVEFORM" begin
    include("test_2D.jl")
    include("test_3D.jl")
    include("test_variable_density.jl")
end
