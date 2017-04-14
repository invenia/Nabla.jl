using AutoGrad2
using Base.Test

srand(123456789)

include("primitive.jl")
include("core.jl")
include("finite_differencing.jl")
include("finite_differencing_test.jl")

# include("sensitivities/scalar.jl")
include("sensitivities/array.jl")
# include("sensitivities/linalg.jl")
# include("sensitivities/blas.jl")

# include("composite/simple.jl")
