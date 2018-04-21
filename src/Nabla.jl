# __precompile__(false)

module Nabla

    using DiffLinearAlgebra, Leibniz, Nullables, Cassette, LinearAlgebra
    import DiffLinearAlgebra: ∇
    using Cassette: @context, @primitive, overdub

    # Some aliases used repeatedly throughout the package.
    export ∇Scalar, ∇Array, ∇ArrayOrScalar, @∇primitive, ∇, ∇all, forward
    const ∇Scalar = Number
    const ∇Array = AbstractArray{<:∇Scalar}
    const ∇ArrayOrScalar = Union{AbstractArray{<:∇Scalar}, ∇Scalar}

    # Code to schedule computations.
    include("core.jl")

    # Lists of definitions of derivatives.
    include("sensitivities/indexing.jl")
    include("sensitivities/scalar.jl")
    include("sensitivities/array.jl")
    include("sensitivities/functional.jl")
    # include("sensitivities/linear_algebra.jl")

end # module Nabla
