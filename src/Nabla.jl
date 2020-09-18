__precompile__()

module Nabla
    using ChainRules
    using ChainRulesCore
    using ExprTools: ExprTools
    using ForwardDiff: ForwardDiff
    using LinearAlgebra
    using Random
    using SpecialFunctions
    using Statistics

    # Some aliases used repeatedly throughout the package.
    export ∇Scalar, ∇Array, SymOrExpr, ∇ArrayOrScalar
    const ∇Scalar = Number
    const ∇Array = AbstractArray{<:∇Scalar}
    const ∇AbstractVector = AbstractVector{<:∇Scalar}
    const ∇AbstractMatrix = AbstractMatrix{<:∇Scalar}
    const ∇ArrayOrScalar = Union{AbstractArray{<:∇Scalar}, ∇Scalar}
    const SymOrExpr = Union{Symbol, Expr}

    # ones/zeros(::AbstractArray) is deprecated in 0.7 and removed in 1.0, but it's a
    # pretty useful method, so we'll define our own for internal use
    for f in (:ones, :zeros)
        like = Symbol(f, "like")
        @eval begin
            $(like)(a::AbstractArray) = $(f)(eltype(a), size(a))
            $(like)(n::Integer) = $(f)(n)
        end
    end

    # Link up to ChainRulesCore so rules are generated when new rrules are declared.
    __init__() = on_new_rule(generate_overload, rrule)

    # Meta-programming utilities specific to Nabla.
    include("code_transformation/util.jl")
    include("code_transformation/differentiable.jl")

    # Functionality for constructing computational graphs.
    include("core.jl")

    # Functionality for defining new sensitivities.
    include("sensitivity.jl")

    # Finite differencing functionality - only used in tests. Would be good to move this
    # into a separate module at some point.
    include("finite_differencing.jl")

    # Sensitivities via ChainRules
    include("sensitivities/chainrules.jl")

    # Sensitivities for the basics.
    include("sensitivities/indexing.jl")
    include("sensitivities/scalar.jl")

    # Sensitivities for functionals.
    include("sensitivities/functional/functional.jl")
    include("sensitivities/functional/reduce.jl")
    include("sensitivities/functional/reducedim.jl")

    # Linear algebra optimisations.
    include("sensitivities/linalg/generic.jl")
    include("sensitivities/linalg/symmetric.jl")
    include("sensitivities/linalg/strided.jl")
    include("sensitivities/linalg/blas.jl")
    include("sensitivities/linalg/diagonal.jl")
    include("sensitivities/linalg/triangular.jl")
    include("sensitivities/linalg/factorization/cholesky.jl")
    include("sensitivities/linalg/factorization/svd.jl")

    # Checkpointing
    include("checkpointing.jl")

end # module Nabla
