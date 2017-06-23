module AutoGrad2

# Some aliases used repeatedly throughout the package.
const SymOrExpr = Union{Symbol, Expr}
const ArrayOrReal = Union{AbstractArray, Real}

export SymOrExpr, ArrayOrReal

# Core functionality.
include("core.jl")
include("sensitivity.jl")

# Reverse-mode sensitivities.
lb, ub = -5., 5.
include("sensitivities/scalar.jl")
# include("sensitivities/functional.jl")
# include("sensitivities/array.jl")
# include("sensitivities/linalg.jl")
# include("sensitivities/blas.jl")

# Demos and examples.
# include("examples/mlp.jl")
# include("examples/vae.jl")

end # module
