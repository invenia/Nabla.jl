using Base.Test, Nabla, Distributions, BenchmarkTools

@testset "Core" begin
    include("core.jl")
    include("code_transformation/util.jl")
    include("code_transformation/differentiable.jl")
    include("sensitivity.jl")
end

@testset "Sensitivities" begin
    include("finite_differencing.jl")

    # Test sensitivities for the basics.
    include("sensitivities/indexing.jl")
    include("sensitivities/scalar.jl")
    include("sensitivities/array.jl")

    # Test sensitivities for functionals.
    @testset "Functional" begin
        include("sensitivities/functional/functional.jl")
        include("sensitivities/functional/reduce.jl")
        include("sensitivities/functional/reducedim.jl")
    end
end
