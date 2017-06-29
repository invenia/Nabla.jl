module AutoGrad2

# Some aliases used repeatedly throughout the package.
const SymOrExpr = Union{Symbol, Expr}
const ArrayOrReal = Union{AbstractArray, Real}

# Set of functions for which we have to use the explicit method-delaration strategy whilst
#  `https://github.com/JuliaLang/julia/issues/22554` is being fixed. May have to continue to
# support this after it is fixed if I wish to continue supporting Julia 0.6 after 0.7 / 1.0
# is released. An alternative strategy is just to maintain an old release version of
# AutoGrad2 which is compatible with Julia-0.6 (similarly for 0.7 / 1.0).
const use_fallback = Set{Symbol}()
push!(use_fallback, :cot)

export SymOrExpr, ArrayOrReal

const pkg_name = current_module()

# Core functionality.
include("core.jl")
include("sensitivity.jl")
# include("sensitivity_sig.jl")

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
