__precompile__(true)

module Nabla

    using DiffLinearAlgebra, FDM, Leibniz, Nullables
    import DiffLinearAlgebra: ∇

    # Some aliases used repeatedly throughout the package.
    export ∇Scalar, ∇Array, SymOrExpr, ∇ArrayOrScalar
    const ∇Scalar = Number
    const ∇Array = AbstractArray{<:∇Scalar}
    const ∇AbstractVector = AbstractVector{<:∇Scalar}
    const ∇AbstractMatrix = AbstractMatrix{<:∇Scalar}
    const ∇ArrayOrScalar = Union{AbstractArray{<:∇Scalar}, ∇Scalar}
    const SymOrExpr = Union{Symbol, Expr}

    # Functionality for constructing computational graphs.
    include("core.jl")
    include("sensitivity.jl")

    # # Finite differencing functionality - only used in tests. Would be good to move this
    # # into a separate module at some point.
    # include("finite_differencing.jl")

    # # Sensitivities for the basics.
    # include("sensitivities/indexing.jl")
    # include("sensitivities/scalar.jl")
    # include("sensitivities/array.jl")

    # Sensitivities for functionals.
    # include("sensitivities/functional/functional.jl")
    # include("sensitivities/functional/reduce.jl")
    # include("sensitivities/functional/reducedim.jl")

    # # Sensitivities for linear algebra optimisations. All imported from DiffLinearAlgebra.
    # include("sensitivities/linear_algebra.jl")

end # module Nabla
