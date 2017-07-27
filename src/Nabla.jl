module Nabla

    srand(1234567897)

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

    # Finite differencing functionality - only used in tests.
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

    # baremodule DiffBase

    #     using ..DiffCore, DualNumbers
    #     import ..DiffCore: ∇, get_original, needs_output

    #     import Base
    #     import Base: include, @inline, @noinline, push!, any, zeros, π, !, method_exists,
    #         error, eltype, zip, similar, size, !=, one, zero, StridedArray, StridedMatrix,
    #         @eval, AbstractMatrix, >, <, ones, eachindex, colon, Val
    #     import Base.Meta.quot

    #     const RealArray = AbstractArray{<:Real}
    #     const RS = StridedMatrix{<:Real}

    #     # Sensitivites for the basics.
    #     include("sensitivities/indexing.jl")
    #     include("sensitivities/scalar.jl")

    #     # Sensitivities for functionals.
    #     include("sensitivities/functional/functional.jl")
    #     include("sensitivities/functional/reduce.jl")
    #     include("sensitivities/functional/reducedim.jl")

    #     # Linear algebra optimisations.
    #     include("sensitivities/linalg/generic.jl")
    #     # include("sensitivities/linalg/uniformscaling.jl")
    #     # include("sensitivities/linalg/diagonal.jl")
    #     # include("sensitivities/linalg/triangular.jl")
    #     include("sensitivities/linalg/strided.jl")

    #     # BLAS sensitivities.
    #     # include("sensitivities/blas.jl")

    # end # module DiffBase

# Demos and examples.
# include("examples/mlp.jl")
# include("examples/vae.jl")

end # module Nabla
