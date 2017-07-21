using Base.Test, Nabla.DiffCore
import Nabla.DiffCore.Leaf

@testset "DiffCore" begin
    include("core.jl")
    include("code_transformation/util.jl")
    include("code_transformation/differentiable.jl")
    include("sensitivity.jl")
end

@differentiable DiffBaseTests begin

    using Base.Test, Distributions
    srand(1234567)

    @testset "DiffBase" begin

        # Check that our finite differencing works well. This should be moved to a separate
        # package at some point. Could there make a concerted effort to have high-quality
        # probabilistic finite differencing (with representation of epistemic uncertainty).
        # In short, would be looking at applying the methodology of probabilistic numerics
        # to gradient estimation through function sampling.
        include("finite_differencing_test.jl")

        # Test sensitivities for the basics.
        include("sensitivities/indexing.jl")
        include("sensitivities/scalar.jl")

        # Test sensitivities for functionals.
        include("sensitivities/functional/functional.jl")
        include("sensitivities/functional/reduce.jl")
        include("sensitivities/functional/reducedim.jl")

        # Test sensitivities for linear algebra optimisations.
        include("sensitivities/linalg/generic.jl")
        # include("sensitivities/linalg/uniformscaling.jl")
        # include("sensitivities/linalg/diagonal.jl")
        # include("sensitivities/linalg/triangular.jl")
        include("sensitivities/linalg/strided.jl")

        # Test BLAS sensitivities.
        # include("sensitivities/blas.jl")
    end
end
