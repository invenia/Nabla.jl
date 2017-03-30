using AutoGrad2
using Base.Test

import Base: sum, sumabs2

srand(123456789)

include("basic_types.jl")
include("primitive.jl")
include("core.jl")
include("finite_differencing.jl")

include("sensitivities/scalar.jl")
include("sensitivities/array.jl")
include("sensitivities/linalg.jl")
include("sensitivities/blas.jl")
