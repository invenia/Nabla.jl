using AutoGrad2
using Base.Test

import AutoGrad2.sensitivity

srand(123456789)

# import Base.sum

# @sensitivity(sum{T<:Union{AbstractArray, Real}}(x::T),
#     [(x̄, x̄ = broadcast!(x->x, similar(x), ȳ), broadcast!(+, x̄, x̄, ȳ))], y, ȳ)

# Code for checking sensitivities.
include("finite_differencing.jl")
include("finite_differencing_test.jl")

# The components of the package.
include("sensitivity.jl")
include("core.jl")

# Sensitivities for individual functions.
include("sensitivities/scalar.jl")
include("sensitivities/array.jl")
# include("sensitivities/linalg.jl")
# include("sensitivities/blas.jl")

# Some compositional functions for integration testing.
# include("composite/simple.jl")
