module AutoGrad2

# Some aliases used repeatedly throughout the package.
const SymOrExpr = Union{Symbol, Expr}
const ArrayOrFloat = Union{AbstractArray, Number}

# Core functionality.
include("core.jl")
include("primitive.jl")
# include("util.jl")

# Reverse-mode sensitivities.
lb, ub = -5., 5.
# include("sensitivities/scalar.jl")
# include("sensitivities/array.jl")
# include("sensitivities/linalg.jl")
# include("sensitivities/blas.jl")

# Demos and examples.
# include("examples/mlp.jl")
# include("examples/vae.jl")

end # module
