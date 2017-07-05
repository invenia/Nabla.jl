using AutoGrad2

import AutoGrad2.∇

add_intercept(:sum, :Base)
add_∇(AutoGrad2.sum, Arg{1}, (p, y, ȳ, x::ArrayOrReal)->broadcast!(x->x, similar(x), ȳ))
add_∇!(AutoGrad2.sum, Arg{1}, (x̄, p, y, ȳ, x::ArrayOrReal)->broadcast!(+, x̄, x̄, ȳ))

@differentiable Tests begin

println(methods(∇))

using Base.Test

srand(123456789)

# @testset "AutoGrad2 tests" begin
begin

    # Code for checking sensitivities.
    include("finite_differencing.jl")
    include("finite_differencing_test.jl")

    # # The components of the package.
    include("core.jl")
    include("sensitivity.jl")

    # # Sensitivities for individual functions.
    include("sensitivities/scalar.jl")
    include("sensitivities/functional.jl")
    # include("sensitivities/array.jl")
    # include("sensitivities/linalg.jl")
    # include("sensitivities/blas.jl")

    # Some compositional functions for integration testing.
    # include("composite/simple.jl")

end

end
