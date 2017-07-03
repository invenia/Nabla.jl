module AutoGrad2

# Some aliases used repeatedly throughout the package.
const SymOrExpr = Union{Symbol, Expr}
const ArrayOrReal = Union{AbstractArray, Real}

export SymOrExpr, ArrayOrReal

# const pkg_name = current_module()

# Core functionality.
include("core.jl")
include("sensitivity.jl")

# Hand code identity because it's really fundamental. It doesn't need to generate a new
# node on the computational graph since it does nothing, but it is useful to have it's
# gradient implemented for use in higher-order functions.
@inline ∇(::typeof(identity), ::Type{Arg{1}}, p, x, y, ȳ) = ȳ

# General reverse-mode sensitivities.
lb, ub = -5., 5.
include("sensitivities/scalar.jl")
# include("sensitivities/functional.jl")
# include("sensitivities/array.jl")
# include("sensitivities/linalg.jl")
# include("sensitivities/blas.jl")

# Demos and examples.
# include("examples/mlp.jl")
# include("examples/vae.jl")

# # Wrap everything from Base which we have not yet created out own versions of in a function
# # whose implementation can be redefined later on, and export it.
# import Base
# base_names = names(Base)
# for name in base_names
#     try
#         if !isdefined(name) && isa(eval(Base, name), Function)
#             @eval AutoGrad2 @noinline ($name)(x...) = Base.$name(x...)
#         else
#             @eval AutoGrad2 import Base.($name)
#         end
#         @eval AutoGrad2 export $name
#     catch
#         push!(base_names, name)
#     end
# end

end # module
