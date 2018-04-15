# __precompile__(false)

module Nabla

    using DiffLinearAlgebra, FDM, Leibniz, Nullables, Cassette, LinearAlgebra
    import DiffLinearAlgebra: ∇
    using Cassette: @context, @primitive, overdub

    # Some aliases used repeatedly throughout the package.
    export ∇Scalar, ∇Array, SymOrExpr, ∇ArrayOrScalar
    const ∇Scalar = Number
    const ∇Array = AbstractArray{<:∇Scalar}
    const ∇AbstractVector = AbstractVector{<:∇Scalar}
    const ∇AbstractMatrix = AbstractMatrix{<:∇Scalar}
    const ∇ArrayOrScalar = Union{AbstractArray{<:∇Scalar}, ∇Scalar}
    const SymOrExpr = Union{Symbol, Expr}

    # Set up context for Cassette.
    @context ∇Ctx

    # Functionality for constructing computational graphs.
    include("core.jl")

    # Sensitivities for the basics.
    include("sensitivities/indexing.jl")
    include("sensitivities/scalar.jl")
    include("sensitivities/array.jl")

    # Sensitivities for functionals.
    include("sensitivities/functional/functional.jl")
    include("sensitivities/functional/reduce.jl")
    # include("sensitivities/functional/reducedim.jl")

    # Sensitivities for linear algebra optimisations. All imported from DiffLinearAlgebra.
    include("sensitivities/linear_algebra.jl")

    # Finite differencing functionality - only used in tests.
    include("finite_differencing.jl")

end # module Nabla
