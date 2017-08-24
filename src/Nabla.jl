module Nabla

    # Some aliases used repeatedly throughout the package.
    export ∇Real, ∇RealArray, SymOrExpr, ArrayOr∇Real
    const ∇Real = Real
    const ∇RealArray = AbstractArray{<:∇Real}
    const ArrayOr∇Real = Union{AbstractArray{<:∇Real}, ∇Real}
    const SymOrExpr = Union{Symbol, Expr}

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

    # Sensitivities for the basics.
    include("sensitivities/indexing.jl")
    include("sensitivities/scalar.jl")

    # Sensitivities for functionals.
    include("sensitivities/functional/functional.jl")
    include("sensitivities/functional/reduce.jl")
    include("sensitivities/functional/reducedim.jl")

    # Linear algebra optimisations.
    include("sensitivities/linalg/generic.jl")
    include("sensitivities/linalg/strided.jl")
    include("sensitivities/linalg/blas.jl")

end # module Nabla
