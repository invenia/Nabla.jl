using Test, Nabla, BenchmarkTools, Cassette, LinearAlgebra, Random, Nullables, DiffRules,
    SpecialFunctions
using Cassette: overdub

# @testset "Core" begin
#     include("core.jl")
# end

@testset "Sensitivities" begin
    include("finite_differencing.jl")

    # Test sensitivities for the basics.
    include("sensitivities/indexing.jl")
    include("sensitivities/scalar.jl")
    include("sensitivities/array.jl")

    # Test sensitivities for functionals.
    @testset "Functional" begin
        include("sensitivities/functional/functional.jl")
        # include("sensitivities/functional/reduce.jl")
        # include("sensitivities/functional/reducedim.jl")
    end
end
