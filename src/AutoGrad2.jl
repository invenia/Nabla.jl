module AutoGrad2

# Core functionality.
include("core.jl")
include("primitive.jl")

# Reverse-mode sensitivities.
include("sensitivities/scalar.jl")
include("sensitivities/array.jl")
include("sensitivities/linalg.jl")
include("sensitivities/blas.jl")

# Demos and examples.
include("examples/mlp.jl")
include("examples/vae.jl")

end # module
