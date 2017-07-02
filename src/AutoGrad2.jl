module AutoGrad2

# Some aliases used repeatedly throughout the package.
const SymOrExpr = Union{Symbol, Expr}
const ArrayOrReal = Union{AbstractArray, Real}

export SymOrExpr, ArrayOrReal

const pkg_name = current_module()

# Core functionality.
include("core.jl")
include("sensitivity.jl")
include("sensitivity_3.jl")

# Hand code identity because it's really fundamental. It doesn't need to generate a new
# node on the computational graph since it does nothing, but it is useful to have it's
# gradient implemented for use in higher-order functions.
@inline ∇(::typeof(identity), ::Type{Arg{1}}, p, x, y, ȳ) = ȳ

# General reverse-mode sensitivities.
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
