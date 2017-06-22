module AutoGrad2

# Some aliases used repeatedly throughout the package.
const SymOrExpr = Union{Symbol, Expr}
const ArrayOrReal = Union{AbstractArray, Real}

# Dictionary logging the expressions for sensitivities w.r.t. each of the parameters of each
# function for which they are explicitly implemented. These are used later on to
# automatically generate efficient implementations of various higher-order functions.
# DO NOT EDIT THIS OR THINGS WILL BREAK!
const _sens_dict = Dict{Int, Symbol}()

# Core functionality.
include("core.jl")
include("sensitivity.jl")

# Reverse-mode sensitivities.
lb, ub = -5., 5.
include("sensitivities/scalar.jl")
include("sensitivities/functional.jl")
# include("sensitivities/array.jl")
# include("sensitivities/linalg.jl")
# include("sensitivities/blas.jl")

# Demos and examples.
# include("examples/mlp.jl")
# include("examples/vae.jl")

end # module
