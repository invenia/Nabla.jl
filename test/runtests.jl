using AutoGrad2
using Base.Test

srand(123456789)

# Code for checking sensitivities.
# include("finite_differencing.jl")
# include("finite_differencing_test.jl")

# The components of the package.
include("primitive.jl")
# include("core.jl")
# include("util.jl")

# Sensitivities for individual functions.
# include("sensitivities/scalar.jl")
# include("sensitivities/array.jl")
# include("sensitivities/linalg.jl")
# include("sensitivities/blas.jl")

# Some compositional functions for integration testing.
# include("composite/simple.jl")
