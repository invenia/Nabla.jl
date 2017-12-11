using Base.Test, Nabla
# using Distributions, BenchmarkTools

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

    # Test sensitivities for linear algebra optimisations.
    @testset "Linear algebra" begin
        include("sensitivities/linalg/generic.jl")
        include("sensitivities/linalg/uniformscaling.jl")
        include("sensitivities/linalg/diagonal.jl")
        include("sensitivities/linalg/triangular.jl")
        include("sensitivities/linalg/strided.jl")
        include("sensitivities/linalg/blas.jl")

        @testset "Factorisations" begin
            include("sensitivities/linalg/factorization/cholesky.jl")
        end
    end
end
