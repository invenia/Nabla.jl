import LinearAlgebra.BLAS: asum, dot, blascopy!, nrm2, scal, scal!, gemm, gemm!, gemv, gemv!,
    syrk, symm, symm!, symv, symv!, trmm, trsm, trmv, trsv, trsv!, ger!

const SA = StridedArray

# # `syrk` sensitivity implementations.
# @explicit_intercepts(
#     syrk,
#     Tuple{Char, Char, ∇Scalar, StridedVecOrMat{<:∇Scalar}},
#     [false, false, true, true],
# )
# function ∇(::typeof(syrk), ::Type{Arg{3}}, p, Y, Ȳ,
#     uplo::Char,
#     trans::Char,
#     α::∇Scalar,
#     A::StridedVecOrMat{<:∇Scalar},
# )
#     g! = uppercase(uplo) == 'L' ? tril! : triu!
#     return sum(g!(Ȳ .* Y)) / α
# end
# function ∇(::typeof(syrk), ::Type{Arg{4}}, p, Y, Ȳ,
#     uplo::Char,
#     trans::Char,
#     α::∇Scalar,
#     A::StridedVecOrMat{<:∇Scalar},
# )
#     triȲ = uppercase(uplo) == 'L' ? tril(Ȳ) : triu(Ȳ)
#     out = gemm('N', trans, α, triȲ .+ triȲ', A)
#     return uppercase(trans) == 'N' ? out : out'
# end
# function ∇(Ā::StridedVecOrMat{T}, ::typeof(syrk), ::Type{Arg{4}}, p, Y, Ȳ,
#     uplo::Char,
#     trans::Char,
#     α::∇Scalar,
#     A::StridedVecOrMat{T},
# ) where T<:∇Scalar
#     triȲ = uppercase(uplo) == 'L' ? tril(Ȳ) : triu(Ȳ)
#     out = gemm('N', trans, α, triȲ .+ triȲ', A)
#     return broadcast!((ā, δā)->ā+δā, Ā, Ā, uppercase(trans) == 'N' ? out : out')
# end

# # `syrk` sensitivity implementations for `α=1`.
# @explicit_intercepts(
#     syrk,
#     Tuple{Char, Char, StridedVecOrMat{<:∇Scalar}},
#     [false, false, true],
# )
# ∇(::typeof(syrk), ::Type{Arg{3}}, p, Y, Ȳ,
#     uplo::Char,
#     trans::Char,
#     A::StridedVecOrMat{<:∇Scalar},
# ) = ∇(syrk, Arg{4}, p, Y, Ȳ, uplo, trans, one(eltype(A)), A)
# ∇(Ā::StridedVecOrMat{T}, ::typeof(syrk), ::Type{Arg{4}}, p, Y, Ȳ,
#     uplo::Char,
#     trans::Char,
#     A::StridedVecOrMat{T},
# ) where T<:∇Scalar = ∇(Ā, syrk, Arg{4}, p, Y, Ȳ, uplo, char, one(eltype(A)), A)

# `symm` sensitivity implementations.
@explicit_intercepts(
    symm,
    Tuple{Char, Char, T, StridedMatrix{T}, StridedMatrix{T}} where T<:∇Scalar,
    [false, false, true, true, true],
)
∇(::typeof(symm), ::Type{Arg{3}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedVecOrMat{T},
) where T<:∇Scalar = sum(Ȳ .* Y) / α
function ∇(::typeof(symm), ::Type{Arg{4}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedVecOrMat{T},
) where T<:∇Scalar
    tmp = uppercase(side) == 'L' ? Ȳ * B' : B'Ȳ
    g! = uppercase(ul) == 'L' ? tril! : triu!
    return α * g!(tmp + tmp' - Diagonal(tmp))
end
function ∇(Ā::StridedMatrix{T}, ::typeof(symm), ::Type{Arg{4}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedVecOrMat{T},
) where T<:∇Scalar
    tmp = uppercase(side) == 'L' ? Ȳ * B' : B'Ȳ
    g! = uppercase(ul) == 'L' ? tril! : triu!
    return broadcast!((ā, δā)->ā + δā, Ā, Ā, α * g!(tmp + tmp' - Diagonal(tmp)))
end
∇(::typeof(symm), ::Type{Arg{5}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedVecOrMat{T},
) where T<:∇Scalar = symm(side, ul, α, A, Ȳ)
∇(B̄::StridedMatrix{T}, ::typeof(symm), ::Type{Arg{5}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedVecOrMat{T},
) where T<:∇Scalar = symm!(side, ul, α, A, Ȳ, 1.0, B̄)

# `symm` sensitivity implementations for `α=1`.
@explicit_intercepts(
    symm,
    Tuple{Char, Char, StridedMatrix{T}, StridedMatrix{T}} where T<:∇Scalar,
    [false, false, true, true],
)
∇(::typeof(symm), ::Type{Arg{3}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    A::StridedMatrix{T},
    B::StridedVecOrMat{T},
) where T<:∇Scalar = ∇(symm, Arg{4}, p, Y, Ȳ, side, ul, one(T), A, B)
∇(Ā::StridedMatrix{T}, ::typeof(symm), ::Type{Arg{3}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    A::StridedMatrix{T},
    B::StridedVecOrMat{T},
) where T<:∇Scalar = ∇(Ā, symm, Arg{4}, p, Y, Ȳ, side, ul, one(T), A, B)
∇(::typeof(symm), ::Type{Arg{4}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    A::StridedMatrix{T},
    B::StridedVecOrMat{T},
) where T<:∇Scalar = ∇(symm, Arg{5}, p, Y, Ȳ, side, ul, one(T), A, B)
∇(B̄::StridedMatrix{T}, ::typeof(symm), ::Type{Arg{4}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    A::StridedMatrix{T},
    B::StridedVecOrMat{T},
) where T<:∇Scalar = ∇(B̄, symm, Arg{5}, p, Y, Ȳ, side, ul, one(T), A, B)

# `symv` sensitivity implementations.
@explicit_intercepts(
    symv,
    Tuple{Char, T, StridedMatrix{T}, StridedVector{T}} where T<:∇Scalar,
    [false, true, true, true],
)
∇(::typeof(symv), ::Type{Arg{2}}, p, y, ȳ,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Scalar = dot(ȳ, y) / α
∇(::typeof(symv), ::Type{Arg{3}}, p, y, ȳ,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Scalar = ∇(symm, Arg{4}, p, y, ȳ, 'L', ul, α, A, x)
∇(Ā::StridedMatrix{T}, ::typeof(symv), ::Type{Arg{3}}, p, y, ȳ,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Scalar = ∇(Ā, symm, Arg{4}, p, y, ȳ, 'L', ul, α, A, x)
∇(::typeof(symv), ::Type{Arg{4}}, p, y, ȳ,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Scalar = symv(ul, α, A, ȳ)
∇(x̄::StridedVector{T}, ::typeof(symv), ::Type{Arg{4}}, p, y, ȳ,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Scalar = symv!(ul, α, A, ȳ, one(T), x̄)

# `symv` sensitivity implementations for `α=1`.
@explicit_intercepts(
    symv,
    Tuple{Char, StridedMatrix{T}, StridedVector{T}} where T<:∇Scalar,
    [false, true, true],
)
∇(::typeof(symv), ::Type{Arg{2}}, p, y, ȳ,
    ul::Char,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Scalar = ∇(symv, Arg{3}, p, y, ȳ, ul, one(T), A, x)
∇(Ā::StridedMatrix{T}, ::typeof(symv), ::Type{Arg{2}}, p, y, ȳ,
    ul::Char,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Scalar = ∇(Ā, symv, Arg{3}, p, y, ȳ, ul, one(T), A, x)
∇(::typeof(symv), ::Type{Arg{3}}, p, y, ȳ,
    ul::Char,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Scalar = ∇(symv, Arg{4}, p, y, ȳ, ul, one(T), A, x)
∇(B̄::StridedVector{T}, ::typeof(symv), ::Type{Arg{3}}, p, y, ȳ,
    ul::Char,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Scalar = ∇(B̄, symv, Arg{4}, p, y, ȳ, ul, one(T), A, x)

# `trmm` sensitivity implementations.
@explicit_intercepts(
    trmm,
    Tuple{Char, Char, Char, Char, T, StridedMatrix{T}, StridedMatrix{T}} where T<:∇Scalar,
    [false, false, false, false, true, true, true],
)
∇(::typeof(trmm), ::Type{Arg{5}}, p, Y, Ȳ,
    side::Char, ul::Char, ta::Char, dA::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedVecOrMat{T},
) where T<:∇Scalar = sum(Ȳ .* Y) / α
function ∇(::typeof(trmm), ::Type{Arg{6}}, p, Y, Ȳ,
    side::Char, ul::Char, ta::Char, dA::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedVecOrMat{T},
) where T<:∇Scalar
    Ā_full = uppercase(side) == 'L' ?
        uppercase(ta) == 'N' ?
            gemm('N', 'T', α, Ȳ, B) :
            gemm('N', 'T', α, B, Ȳ) :
        uppercase(ta) == 'N' ?
            gemm('T', 'N', α, B, Ȳ) :
            gemm('T', 'N', α, Ȳ, B)
    dA == 'U' && fill!(view(Ā_full, diagind(Ā_full)), zero(T))
    return (uppercase(ul) == 'L' ? tril! : triu!)(Ā_full)
end
∇(::typeof(trmm), ::Type{Arg{7}}, p, Y, Ȳ,
    side::Char, ul::Char, ta::Char, dA::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedVecOrMat{T},
) where T<:∇Scalar =
    uppercase(side) == 'L' ?
        uppercase(ta) == 'N' ?
            trmm('L', ul, 'T', dA, α, A, Ȳ) :
            trmm('L', ul, 'N', dA, α, A, Ȳ) :
        uppercase(ta) == 'N' ?
            trmm('R', ul, 'T', dA, α, A, Ȳ) :
            trmm('R', ul, 'N', dA, α, A, Ȳ)

# `trmv` sensitivity implementations.
@explicit_intercepts(
    trmv,
    Tuple{Char, Char, Char, StridedMatrix{T}, StridedVector{T}} where T<:∇Scalar,
    [false, false, false, true, true],
)
function ∇(::typeof(trmv), ::Type{Arg{4}}, p, y, ȳ,
    ul::Char, ta::Char, dA::Char,
    A::StridedMatrix{T},
    b::StridedVector{T},
) where T<:∇Scalar
    Ā = (uppercase(ul) == 'L' ? tril! : triu!)(uppercase(ta) == 'N' ? ȳ * b' : b * ȳ')
    dA == 'U' && fill!(view(Ā, diagind(Ā)), zero(T))
    return Ā
end
∇(::typeof(trmv), ::Type{Arg{5}}, p, y, ȳ,
    ul::Char, ta::Char, dA::Char,
    A::StridedMatrix{T},
    b::StridedVector{T},
) where T<:∇Scalar = trmv(ul, uppercase(ta) == 'N' ? 'T' : 'N', dA, A, ȳ)

# `trsm` sensitivity implementations.
@explicit_intercepts(
    trsm,
    Tuple{Char, Char, Char, Char, T, StridedMatrix{T}, StridedMatrix{T}} where T<:∇Scalar,
    [false, false, false, false, true, true, true],
)
∇(::typeof(trsm), ::Type{Arg{5}}, p, Y, Ȳ,
    side::Char, ul::Char, ta::Char, dA::Char,
    α::T,
    A::StridedMatrix{T},
    X::StridedMatrix{T},
) where T<:∇Scalar = sum(Ȳ .* Y) / α
function ∇(::typeof(trsm), ::Type{Arg{6}}, p, Y, Ȳ,
    side::Char, ul::Char, ta::Char, dA::Char,
    α::T,
    A::StridedMatrix{T},
    X::StridedVecOrMat{T},
) where T<:∇Scalar
    Ā_full = uppercase(side) == 'L' ?
        uppercase(ta) == 'N' ?
            trsm('L', ul, 'T', dA, -1.0, A, Ȳ * Y') :
            trsm('R', ul, 'T', dA, -1.0, A, Y * Ȳ') :
        uppercase(ta) == 'N' ?
            trsm('R', ul, 'T', dA, -1.0, A, Y'Ȳ) :
            trsm('L', ul, 'T', dA, -1.0, A, Ȳ'Y)
    dA == 'U' && fill!(view(Ā_full, diagind(Ā_full)), zero(T))
    return (uppercase(ul) == 'L' ? tril! : triu!)(Ā_full)
end
∇(::typeof(trsm), ::Type{Arg{7}}, p, Y, Ȳ,
    side::Char, ul::Char, ta::Char, dA::Char,
    α::T,
    A::StridedMatrix{T},
    X::StridedMatrix{T},
) where T<:∇Scalar =
    uppercase(side) == 'L' ?
        uppercase(ta) == 'N' ?
            trsm('L', ul, 'T', dA, α, A, Ȳ) :
            trsm('L', ul, 'N', dA, α, A, Ȳ) :
        uppercase(ta) == 'N' ?
            trsm('R', ul, 'T', dA, α, A, Ȳ) :
            trsm('R', ul, 'N', dA, α, A, Ȳ)

# `trsv` sensitivity implementations.
@explicit_intercepts(
    trsv,
    Tuple{Char, Char, Char, StridedMatrix{T}, StridedVector{T}} where T<:∇Scalar,
    [false, false, false, true, true],
)
function ∇(::typeof(trsv), ::Type{Arg{4}}, p, y, ȳ,
    ul::Char, ta::Char, dA::Char,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Scalar
    Ā = ∇(trsm, Arg{6}, p, y, ȳ, 'L', ul, ta, dA, one(T), A, x)
    dA == 'U' && fill!(view(Ā, diagind(Ā)), zero(T))
    return Ā
end
∇(::typeof(trsv), ::Type{Arg{5}}, p, y, ȳ,
    ul::Char, ta::Char, dA::Char,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Scalar = trsv(ul, uppercase(ta) == 'N' ? 'T' : 'N', dA, A, ȳ)

# # # TODO: Banded matrix operations.
# # # gbmv
# # # sbmv
