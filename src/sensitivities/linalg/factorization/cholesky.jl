import LinearAlgebra.BLAS: gemv, gemv!, gemm!, trsm!, axpy!, ger!
import LinearAlgebra: cholesky, Cholesky
import Base: getproperty

Base.@deprecate chol(X) cholesky(X).U


const AM = AbstractMatrix
const UT = UpperTriangular


@explicit_intercepts(
    Cholesky,
    Tuple{AbstractMatrix{<:∇Scalar}, Union{Char, Symbol}, Integer},
    [true, false, false],
)

function ∇(
    ::Type{Cholesky},
    ::Type{Arg{1}},
    p,
    C::Cholesky,
    X̄::Union{UpperTriangular, LowerTriangular},
    X::Union{UpperTriangular, LowerTriangular},
    uplo::Union{Char, Symbol},
    info::Integer,
)
    # We aren't doing any actual computation if we've constructed a Cholesky object
    # directly, so just pass through this call and return the sensitivies
    return X̄
end
function ∇(
    ::Type{Cholesky},
    ::Type{Arg{1}},
    p,
    C::Cholesky,
    X̄::Tangent{<:Cholesky},
    X::Union{UpperTriangular, LowerTriangular},
    uplo::Union{Char, Symbol},
    info::Integer,
)
    return getproperty(X̄, Symbol(uplo))
end

# Yar, some work arounds for breaking changes in ChainRules.jl
# https://github.com/JuliaDiff/ChainRules.jl/pull/630

# Single arg function was dropped
function ChainRules.rrule(::typeof(cholesky), A::AbstractMatrix{<:Real})
    return ChainRules.rrule(cholesky, A, Val(false))
end

# U and L properties were replaced with factors
# This should probably be moved to ChainRules to support both options.
function Base.getproperty(tangent::Tangent{P, T}, sym::Symbol) where {P <: Cholesky, T <: NamedTuple}
    idx = hasfield(T, :factors) && sym in (:U, :L) ? :factors : sym
    hasfield(T, idx) || return ZeroTangent()
    return unthunk(getfield(ChainRulesCore.backing(tangent), idx))
end
