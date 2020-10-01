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
    X̄::Composite{<:Cholesky},
    X::Union{UpperTriangular, LowerTriangular},
    uplo::Union{Char, Symbol},
    info::Integer,
)
    return getproperty(X̄, Symbol(uplo))
end
