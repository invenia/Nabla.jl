using Revise
using Test, Nabla, Cassette, LinearAlgebra, Random, Nullables, DiffRules, SpecialFunctions
import Nabla: init_rvs_tape, preprocess, forward

@testset "Nabla" begin

    # include("core.jl")

    # Utility functionality for use in tests.
    include("finite_differencing.jl")
    # include("finite_differencing_tests.jl")

    # Test sensitivities for the basics.
    # include("sensitivities/indexing.jl")
    # include("sensitivities/scalar.jl")
    # include("sensitivities/array.jl")
    include("sensitivities/functional.jl")
    include("sensitivities/linear_algebra.jl")
end
