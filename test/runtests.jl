using Nabla
using Test, LinearAlgebra, Statistics, Random, ForwardDiff
using Distributions, BenchmarkTools, SpecialFunctions

using Nabla: unbox, pos, tape, oneslike, zeroslike

# Helper function for comparing `Ref`s, since they don't compare equal under `==`
ref_equal(a::Ref{T}, b::Ref{T}) where {T} = a[] == b[]
ref_equal(a::Ref, b::Ref) = false

# for comparing against scalar rules
derivative_via_frule(f, x) = last(Nabla.frule((Nabla.NO_FIELDS, 1.0), f, x))
# Sensiblity checkes that his is defined right
@test derivative_via_frule(cos, 0) == 0
@test derivative_via_frule(sin, 0) == 1
@test derivative_via_frule(sin, 1.2) == derivative_via_frule(sin, 2π + 1.2)

# These are the core scalar sensitives Nabla expects to have defined
# we test against them both for sensitives/scalar.jl and in sensitivities/functional.jl

const UNARY_SCALAR_SENSITIVITIES = [
    # Base:
    +, -, abs, abs2, acos, acosd, acosh, acot, acotd, acoth, acsc, acscd, acsch, asec,
    asecd, asech, asin, asind, asinh, atand, atanh, cbrt, cos, cosd, cosh, cospi, cot,cotd,
    coth, csc, cscd, csch, deg2rad, exp, exp10, exp2, expm1, inv, log, log10, log2,
    rad2deg, sec, secd, sech, sin, sind, sinh, sinpi, sqrt, tan, tand, tanh, transpose,
    # SpecialFunctions.jl:
    airyai, airyaiprime, airybi, airybiprime, besselj0, besselj1, bessely0, bessely1,
    dawson, digamma, erf, erfc, erfcinv, erfcx, erfi, erfinv, gamma, invdigamma, lgamma,
    trigamma,
]

const BINARY_SCALAR_SENSITIVITIES = [
    # Base:
    *, +, -, /, \, ^, hypot, max, min,
    # SpecialFunctions.jl:
    besseli, besselj, besselk, bessely, beta, lbeta, polygamma,
]

const ONLY_DIFF_IN_SECOND_ARG_SENSITIVITIES = [
    besseli, besselj, besselk, bessely, polygamma
]

@testset "Nabla.jl" begin

@testset "Core" begin
    include("core.jl")
    include("code_transformation/util.jl")
    include("code_transformation/differentiable.jl")
    include("sensitivity.jl")
end

@testset "Sensitivities" begin
    include("finite_differencing.jl")

    # Test sensitivities for the basics.
    include("sensitivities/indexing.jl")
    include("sensitivities/scalar.jl")
    include("sensitivities/array.jl")

    # Test sensitivities for functionals.
    @testset "Functional" begin
        include("sensitivities/functional/functional.jl")
        include("sensitivities/functional/reduce.jl")
        include("sensitivities/functional/reducedim.jl")
    end

    # Test sensitivities for linear algebra optimisations.
    @testset "Linear algebra" begin
        include("sensitivities/linalg/generic.jl")
        include("sensitivities/linalg/symmetric.jl")
        include("sensitivities/linalg/uniformscaling.jl")
        include("sensitivities/linalg/diagonal.jl")
        include("sensitivities/linalg/triangular.jl")
        include("sensitivities/linalg/strided.jl")
        include("sensitivities/linalg/blas.jl")

        @testset "Factorisations" begin
            include("sensitivities/linalg/factorization/cholesky.jl")
            include("sensitivities/linalg/factorization/svd.jl")
        end
    end

    include("checkpointing.jl")
end

end
