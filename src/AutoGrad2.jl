module AutoGrad2

# Core functionality.
include("basic_types.jl")
include("core.jl")
include("primitive.jl")
include("finite_differencing.jl")

# Reverse-mode sensitivities.
include("sensitivities/scalar.jl")
include("sensitivities/array.jl")
include("sensitivities/linalg.jl")
include("sensitivities/blas.jl")

# Demos and examples.
include("examples/mlp.jl")

end # module
