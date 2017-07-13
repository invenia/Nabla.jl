module Nabla

    module DiffCore

        # Some aliases used repeatedly throughout the package.
        export SymOrExpr, ArrayOrReal
        const SymOrExpr = Union{Symbol, Expr}
        const ArrayOrReal = Union{AbstractArray{T} where T<:Real, Real}

        # Functionality for constructing computational graphs.
        include("core.jl")

        # Functionality for defining new sensitivities.
        include("sensitivity.jl")

        # Functionality to create a baremodule which 
        include("differentiable.jl")

        # Finite differencing functionality for defining tests.
        include("finite_differencing.jl")

    end # module Core

    baremodule DiffBase

        using ..DiffCore
        import ..DiffCore: ∇, get_original

        import Base
        import Base: include, @inline, @noinline, push!, any, zeros, π, !, method_exists,
            error, eltype, zip, similar, size, !=, one, zero, StridedArray, StridedMatrix,
            @eval, AbstractMatrix, >, <, ones
        import Base.Meta.quot

        const RealArray = AbstractArray{T} where T<:Real
        const RS = StridedMatrix{T} where T<:Real

        # Sensitivites for the basics.
        include("sensitivities/indexing.jl")
        include("sensitivities/scalar.jl")
        include("sensitivities/functional.jl")

        # Linear algebra optimisations.
        include("sensitivities/linalg/generic.jl")
        # include("sensitivities/linalg/uniformscaling.jl")
        # include("sensitivities/linalg/diagonal.jl")
        # include("sensitivities/linalg/triangular.jl")
        include("sensitivities/linalg/strided.jl")

        # BLAS sensitivities.
        # include("sensitivities/blas.jl")

    end # module DiffCore

# Demos and examples.
# include("examples/mlp.jl")
# include("examples/vae.jl")

end # module Nabla
