import Base.LinAlg.BLAS: asum, dot, blascopy!, nrm2, scal, scal!, gemm, gemv, syrk, symm,
    symv, trmm, trsm, trmv, trsv

const SA = StridedArray

# Short-form `dot`.
@explicit_intercepts dot Tuple{StridedArray, StridedArray}
∇(::typeof(dot), ::Type{Arg{1}}, p, z, z̄, x::SA, y::SA) = z̄ .* y
∇(::typeof(dot), ::Type{Arg{2}}, p, z, z̄, x::SA, y::SA) = z̄ .* x
∇(x̄, ::typeof(dot), ::Type{Arg{1}}, p, z, z̄, x::SA, y::SA) = (x̄ .= x̄ .+ z̄ .* y)
∇(ȳ, ::typeof(dot), ::Type{Arg{2}}, p, z, z̄, x::SA, y::SA) = (ȳ .+ ȳ .+ z̄ .* x)

# Long-form `dot`.
@explicit_intercepts(
    dot,
    Tuple{Int, StridedArray, Int, StridedArray, Int},
    [false, true, false, true, false],
)
∇(::typeof(dot), ::Type{Arg{2}}, p, z, z̄, n::Int, x::SA, ix::Int, y::SA, iy::Int) =
    scal!(n, z̄, blascopy!(n, y, iy, zeros(x), ix), ix)
∇(::typeof(dot), ::Type{Arg{4}}, p, z, z̄, n::Int, x::SA, ix::Int, y::SA, iy::Int) =
    scal!(n, z̄, blascopy!(n, x, ix, zeros(y), iy), iy)
∇(x̄, ::typeof(dot), ::Type{Arg{2}}, p, z, z̄, n::Int, x::SA, ix::Int, y::SA, iy::Int) =
    (x̄ .= x̄ .+ scal!(n, z̄, blascopy!(n, y, iy, zeros(x), ix), ix))
∇(ȳ, ::typeof(dot), ::Type{Arg{4}}, p, z, z̄, n::Int, x::SA, ix::Int, y::SA, iy::Int) =
    (ȳ .= ȳ .+ scal!(n, z̄, blascopy!(n, x, ix, zeros(y), iy), iy))

# Short-form `nrm2`.
@explicit_intercepts nrm2 Tuple{Union{StridedVector, Array}}
∇(::typeof(nrm2), ::Type{Arg{1}}, p, y, ȳ, x) = x * (ȳ / y)
∇(x̄, ::typeof(nrm2), ::Type{Arg{1}}, p, y, ȳ, x) = (x̄ .= x̄ .+ x .* (ȳ / y))

# Long-form `nrm2`.
@explicit_intercepts(
    nrm2,
    Tuple{Integer, Union{DenseArray, Ptr{<:AbstractFloat}}, Integer},
    [false, true, false],
)
∇(::typeof(nrm2), ::Type{Arg{2}}, p, y, ȳ, n::Integer, x, inc::Integer) =
    scal!(n, ȳ / y, blascopy!(n, x, inc, zeros(x), inc), inc)
∇(x̄, ::typeof(nrm2), ::Type{Arg{2}}, p, y, ȳ, n::Integer, x, inc::Integer) =
    (x̄ .= x̄ .+ scal!(n, ȳ / y, blascopy!(n, x, inc, zeros(x), inc), inc))

# Short-form `asum`.
@explicit_intercepts asum Tuple{Union{StridedVector, Array}}
∇(::typeof(asum), ::Type{Arg{1}}, p, y, ȳ, x) = ȳ .* sign.(x)
∇(x̄, ::typeof(asum), ::Type{Arg{1}}, p, y, ȳ, x) = (x̄ .= x̄ .+ x .* ȳ .* sign.(x))

# Long-form `asum`.
@explicit_intercepts(
    asum,
    Tuple{Integer, Union{DenseArray, Ptr{<:AbstractFloat}}, Integer},
    [false, true, false],
)
∇(::typeof(asum), ::Type{Arg{2}}, p, y, ȳ, n::Integer, x, inc::Integer) =
    scal!(n, ȳ, blascopy!(n, sign.(x), inc, zeros(x), inc), inc)
∇(x̄, ::typeof(asum), ::Type{Arg{2}}, p, y, ȳ, n::Integer, x, inc::Integer) =
    (x̄ .= x̄ .+ scal!(n, ȳ, blascopy!(n, sign.(x), inc, zeros(x), inc), inc))


# Some weird stuff going on that I haven't figured out yet.
# let f = :(scal{T <: AbstractArray, V <: AbstractFloat})
#     ā = :(blascopy!(n, z̄, inc, zeros(X), inc) .* X)
#     X̄ = :(scal!(n, a, z̄, inc))
#     @eva; @primitive $f(n::Int, a::V, X::T, inc::Int) z z̄ false $ā $X̄ false
# end

# `gemm` sensitivities implementation.
@explicit_intercepts(
    gemm,
    Tuple{Char, Char, StridedMatrix{T}, StridedMatrix{T}} where T<:∇Real,
    [false, false, true, true],
)
∇(::typeof(gemm), ::Type{Arg{3}}, p, Y, Ȳ,
    tA::Char,
    tB::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where T<:∇Real = sum(Ȳ .* Y) / α

∇(::typeof(gemm), ::Type{Arg{4}}, p, Y, Ȳ,
    tA::Char,
    tB::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where T<:∇Real =
    uppercase(tA) == 'N' ?
        uppercase(tB) == 'N' ?
            gemm('N', 'T', α, Ȳ, B) :
            gemm('N', 'N', α, Ȳ, B) :
        uppercase(tB) == 'N' ?
            gemm('N', 'T', α, B, Ȳ) :
            gemm('T', 'T', α, B, Ȳ)

∇(Ā::StridedMatrix{T}, ::typeof(gemm), ::Type{Arg{4}}, _, Y, Ȳ,
    tA::Char,
    tB::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where T<:∇Real =
    uppercase(tA) == 'N' ?
        uppercase(tB) == 'N' ?
            gemm!('N', 'T', α, Ȳ, B, 1.0, Ā) :
            gemm!('N', 'N', α, Ȳ, B, 1.0, Ā) :
        uppercase(tB) == 'N' ?
            gemm!('N', 'T', α, B, Ȳ, 1.0, Ā) :
            gemm!('T', 'T', α, B, Ȳ, 1.0, Ā)

∇(::typeof(gemm), ::Type{Arg{5}}, p, Y, Ȳ,
    tA::Char,
    tB::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where T<:∇Real =
    uppercase(tA) == 'N' ?
        uppercase(tB) == 'N' ?
            gemm('T', 'N', α, A, Ȳ) :
            gemm('T', 'N', α, Ȳ, A) :
        uppercase(tB) == 'N' ?
            gemm('N', 'N', α, A, Ȳ) :
            gemm('T', 'T', α, Ȳ, A)

∇(B̄::StridedMatrix{T}, ::typeof(gemm), ::Type{Arg{5}}, _, Y, Ȳ,
    tA::Char,
    tB::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where T<:∇Real =
    uppercase(tA) == 'N' ?
        uppercase(tB) == 'N' ?
            gemm!('T', 'N', α, A, Ȳ, 1.0, B̄) :
            gemm!('T', 'N', α, Ȳ, A, 1.0, B̄) :
        uppercase(tB) == 'N' ?
            gemm!('N', 'N', α, A, Ȳ, 1.0, B̄) :
            gemm!('T', 'T', α, Ȳ, A, 1.0, B̄)

# `gemm` sensitivities implementation for `α = 1`.
@explicit_intercepts(
    gemm,
    Tuple{Char, Char, T, StridedMatrix{T}, StridedMatrix{T}} where T<:∇Real,
    [false, false, true, true, true]
)
∇(::typeof(gemm), ::Type{Arg{3}}, p, Y, Ȳ,
    tA::Char,
    tB::Char,
    A::StridedMatrix{T},
    B::StridedMatrix{T}
) where T<:∇Real = ∇(gemm, Arg{4}, p, Y, Ȳ, tA, tB, one(T), A, B)
∇(Ā, ::typeof(gemm), ::Type{Arg{3}}, p, Y, Ȳ,
    tA::Char,
    tB::Char,
    A::StridedMatrix{T},
    B::StridedMatrix{T}
) where T<:∇Real = ∇(Ā, gemm, Arg{4}, p, Y, Ȳ, tA, tB, one(T), A, B)
∇(::typeof(gemm), ::Type{Arg{4}}, p, Y, Ȳ,
    tA::Char,
    tB::Char,
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where T<:∇Real = ∇(gemm, Arg{5}, p, Y, Ȳ, tA, tB, one(T), A, B)
∇(B̄, ::typeof(gemm), ::Type{Arg{4}}, p, Y, Ȳ,
    tA::Char,
    tB::Char,
    A::StridedMatrix{T},
    B::StridedMatrix{T}
) where T<:∇Real = ∇(B̄, gemm, Arg{5}, p, Y, Ȳ, tA, tB, one(T), A, B)

# `gemv` sensitivities implementation.
@explicit_intercepts(
    gemv,
    Tuple{Char, T, StridedMatrix{T}, StridedVector{T}} where T<:∇Real,
    [false, true, true, true],
)
∇(::typeof(gemv), ::Type{Arg{2}}, p, y, ȳ,
    tA::Char,
    α::T,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Real = dot(ȳ, y) / α
∇(::typeof(gemv), ::Type{Arg{3}}, p, y, ȳ,
    tA::Char,
    α::T,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Real = uppercase(tA) == 'N' ? α * ȳ * x.' : α * x * ȳ.'
∇(Ā::StridedMatrix{T}, ::typeof(gemv), ::Type{Arg{3}}, _, y, ȳ,
    tA::Char,
    α::T,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Real = uppercase(tA) == 'N' ? ger!(α, ȳ, x, Ā) : ger!(α, x, ȳ, Ā)
∇(::typeof(gemv), ::Type{Arg{4}}, p, y, ȳ,
    tA::Char,
    α::T,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Real = gemv(uppercase(tA) == 'N' ? 'T' : 'N', α, A, ȳ)
∇(x̄::StridedVector{T}, ::typeof(gemv), ::Type{Arg{4}}, _, y, ȳ,
    tA::Char,
    α::T,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Real = gemv!(uppercase(tA) == 'N' ? 'T' : 'N', α, A, ȳ, one(T), x̄)

# `gemv` sensitivities implementation with `α = 1`.
@explicit_intercepts(
    gemv,
    Tuple{Char, StridedMatrix{T}, StridedVector{T}} where T<:∇Real,
    [false, true, true],
)
∇(::typeof(gemv), ::Type{Arg{2}}, p, y, ȳ,
    tA::Char,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Real = ∇(gemv, Arg{3}, p, y, ȳ, tA, one(T), A, x)
∇(Ā::StridedMatrix{T}, ::typeof(gemv), ::Type{Arg{2}}, p, y, ȳ,
    tA::Char,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Real = ∇(Ā, gemv, Arg{3}, p, y, ȳ, tA, one(T), A, x)
∇(::typeof(gemv), ::Type{Arg{3}}, p, y, ȳ,
    tA::Char,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Real = ∇(gemv, Arg{4}, p, y, ȳ, tA, one(T), A, x)
∇(x̄::StridedVector{T}, ::typeof(gemv), ::Type{Arg{3}}, p, y, ȳ,
    tA::Char,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Real = ∇(x̄, gemv, Arg{4}, p, y, ȳ, tA, one(T), A, x)

# `syrk` sensitivity implementations.
@explicit_intercepts(
    syrk,
    Tuple{Char, Char, ∇Real, StridedVecOrMat{<:∇Real}},
    [false, false, true, true],
)
function ∇(::typeof(syrk), ::Type{Arg{3}}, p, Y, Ȳ,
    uplo::Char,
    trans::Char,
    α::∇Real,
    A::StridedVecOrMat{<:∇Real},
)
    g! = uppercase(uplo) == 'L' ? tril! : triu!
    return sum(g!(Ȳ .* Y)) / α
end
function ∇(::typeof(syrk), ::Type{Arg{4}}, p, Y, Ȳ,
    uplo::Char,
    trans::Char,
    α::∇Real,
    A::StridedVecOrMat{<:∇Real},
)
    triȲ = uppercase(uplo) == 'L' ? tril(Ȳ) : triu(Ȳ)
    out = gemm('N', trans, α, triȲ .+ triȲ.', A)
    return uppercase(trans) == 'N' ? out : out.'
end
function ∇(Ā::StridedVecOrMat{T}, ::typeof(syrk), ::Type{Arg{4}}, p, Y, Ȳ,
    uplo::Char,
    trans::Char,
    α::∇Real,
    A::StridedVecOrMat{T},
) where T<:∇Real
    triȲ = uppercase(uplo) == 'L' ? tril(Ȳ) : triu(Ȳ)
    out = gemm('N', trans, α, triȲ .+ triȲ.', A)
    return broadcast!((ā, δā)->ā+δā, Ā, Ā, uppercase(trans) == 'N' ? out : out.')
end

# `syrk` sensitivity implementations for `α=1`.
@explicit_intercepts(
    syrk,
    Tuple{Char, Char, StridedVecOrMat{<:∇Real}},
    [false, false, true],
)
∇(::typeof(syrk), ::Type{Arg{3}}, p, Y, Ȳ,
    uplo::Char,
    trans::Char,
    A::StridedVecOrMat{<:∇Real},
) = ∇(syrk, Arg{4}, p, Y, Ȳ, uplo, trans, one(eltype(A)), A)
∇(Ā::StridedVecOrMat{T}, ::typeof(syrk), ::Type{Arg{4}}, p, Y, Ȳ,
    uplo::Char,
    trans::Char,
    A::StridedVecOrMat{T},
) where T<:∇Real = ∇(Ā, syrk, Arg{4}, p, Y, Ȳ, uplo, char, one(eltype(A)), A)

# `symm` sensitivity implementations.
@explicit_intercepts(
    symm,
    Tuple{Char, Char, T, StridedMatrix{T}, StridedMatrix{T}} where T<:∇Real,
    [false, false, true, true, true],
)
∇(::typeof(symm), ::Type{Arg{3}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where T<:∇Real = sum(Ȳ .* Y) / α
function ∇(::typeof(symm), ::Type{Arg{4}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where T<:∇Real
    tmp = uppercase(side) == 'L' ? Ȳ * B.' : B.'Ȳ
    g! = uppercase(ul) == 'L' ? tril! : triu!
    return α * g!(tmp + tmp' - Diagonal(tmp))
end
function ∇(Ā::StridedMatrix{T}, ::typeof(symm), ::Type{Arg{4}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where T<:∇Real
    tmp = uppercase(side) == 'L' ? Ȳ * B.' : B.'Ȳ
    g! = uppercase(ul) == 'L' ? tril! : triu!
    return broadcast!((ā, δā)->ā + δā, Ā, Ā, α * g!(tmp + tmp' - Diagonal(tmp)))
end
∇(::typeof(symm), ::Type{Arg{5}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where T<:∇Real = symm(side, ul, α, A, Ȳ)
∇(B̄::StridedMatrix{T}, ::typeof(symm), ::Type{Arg{5}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where T<:∇Real = symm!(side, ul, α, A, Ȳ, 1.0, B̄)

# `symm` sensitivity implementations for `α=1`.
@explicit_intercepts(
    symm,
    Tuple{Char, Char, StridedMatrix{T}, StridedMatrix{T}} where T<:∇Real,
    [false, false, true, true],
)
∇(::typeof(symm), ::Type{Arg{3}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where T<:∇Real = ∇(symm, Arg{4}, p, Y, Ȳ, side, ul, one(T), A, B)
∇(Ā::StridedMatrix{T}, ::typeof(symm), ::Type{Arg{3}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where T<:∇Real = ∇(Ā, symm, Arg{4}, p, Y, Ȳ, side, ul, one(T), A, B)
∇(::typeof(symm), ::Type{Arg{4}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where T<:∇Real = ∇(symm, Arg{5}, p, Y, Ȳ, side, ul, one(T), A, B)
∇(B̄::StridedMatrix{T}, ::typeof(symm), ::Type{Arg{4}}, p, Y, Ȳ,
    side::Char,
    ul::Char,
    A::StridedMatrix{T},
    B::StridedMatrix{T},
) where T<:∇Real = ∇(B̄, symm, Arg{5}, p, Y, Ȳ, side, ul, one(T), A, B)

# `symv` sensitivity implementations.
@explicit_intercepts(
    symv,
    Tuple{Char, T, StridedMatrix{T}, StridedVector{T}} where T<:∇Real,
    [false, true, true, true],
)
∇(::typeof(symv), ::Type{Arg{2}}, p, y, ȳ,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Real = dot(ȳ, y) / α
∇(::typeof(symv), ::Type{Arg{3}}, p, y, ȳ,
    ul::Char,
    α::T,
    A::StridedMatrix{T},
    x::StridedVector{T},
) where T<:∇Real

# `symv` sensitivity implementations for `α=1`.
@explicit_intercepts(
    symv,
    Tuple{Char, T, StridedMatrix{T}, StridedVector{T}} where T<:∇Real,
    [false, true, true, true],
)

# # symv
# let f = :symv
#     ᾱ, ᾱ! = :(ᾱ = dot(ȳ, y) / α), :(ᾱ += dot(ȳ, y) / α)
#     Ā, Ā! = :(Ā = Ā_symm(ȳ, 'L', ul, α, A, x)), :(Ā!_symm(ȳ, 'L', ul, α, A, x, Ā))
#     x̄, x̄! = :(x̄ = symv(ul, α, A, ȳ)), :(symv!(ul, α, A, ȳ, 1.0, x̄))
#     generate_primitive(f,
#         [:(T <: AbstractFloat), :(V <: StridedMatrix), :(W <: StridedVector)],
#         [:ul, :α, :A, :x], [:nothing, :ᾱ, :Ā, :x̄], [:Char, :T, :V, :W],
#         [false, true, true, true], :y, :ȳ,
#         [:nothing, ᾱ, Ā, x̄], [:nothing, ᾱ!, Ā!, x̄!])
# end

# # trmm
# function Ā_trmm(Y, Ȳ, side, ul, ta, dA, α, A, B)
#     Ā_full = uppercase(side) == 'L' ?
#         uppercase(ta) == 'N' ?
#             gemm('N', 'T', α, Ȳ, B) :
#             gemm('N', 'T', α, B, Ȳ) :
#         uppercase(ta) == 'N' ?
#             gemm('T', 'N', α, B, Ȳ) :
#             gemm('T', 'N', α, Ȳ, B)
#     return (uppercase(ul) == 'L' ? tril! : triu!)(Ā_full)
# end
# function B̄_trmm(Y, Ȳ, side, ul, ta, dA, α, A, B)
#     uppercase(side) == 'L' ?
#         uppercase(ta) == 'N' ?
#             trmm('L', ul, 'T', dA, α, A, Ȳ) :
#             trmm('L', ul, 'N', dA, α, A, Ȳ) :
#         uppercase(ta) == 'N' ?
#             trmm('R', ul, 'T', dA, α, A, Ȳ) :
#             trmm('R', ul, 'N', dA, α, A, Ȳ)
# end
# let f = :trmm
#     ᾱ, ᾱ! = :(ᾱ = sum(Ȳ .* Y) / α), :(ᾱ += sum(Ȳ .* Y) / α)
#     Ā = :(Ā = Ā_trmm(Y, Ȳ, side, ul, ta, dA, α, A, B))
#     Ā! = :(broadcast!((ā, δā)->ā + δā, Ā, Ā, Ā_trmm(Y, Ȳ, side, ul, ta, dA, α, A, B)))
#     B̄ = :(B̄ = B̄_trmm(Y, Ȳ, side, ul, ta, dA, α, A, B))
#     B̄! = :(broadcast!((b̄, δb̄)->b̄ + δb̄, B̄, B̄, B̄_trmm(Y, Ȳ, side, ul, ta, dA, α, A, B)))
#     generate_primitive(f, [:(T <: Number), :(V <: StridedMatrix), :(W <: StridedMatrix)],
#         [:side, :ul, :ta, :dA, :α, :A, :B],
#         [:nothing, :nothing, :nothing, :nothing, :ᾱ, :Ā, :B̄],
#         [:Char, :Char, :Char, :Char, :T, :V, :W],
#         [false, false, false, false, true, true, true], :Y, :Ȳ,
#         [:nothing, :nothing, :nothing, :nothing, ᾱ, Ā, B̄],
#         [:nothing, :nothing, :nothing, :nothing, ᾱ!, Ā!, B̄!])
# end

# # trsv
# let f = :trmv
#     Āc = :((uppercase(ul) == 'L' ? tril! : triu!)(uppercase(ta) == 'N' ? ȳ * b.' : b * ȳ.'))
#     Ā, Ā! = :(Ā = $Āc), :(broadcast!((ā, δā)->ā + δā, Ā, Ā, Āc))
#     b̄ = :(b̄ = trmv(ul, uppercase(ta) == 'N' ? 'T' : 'N', dA, A, ȳ))
#     b̄! = :(trmv!(ul, uppercase(ta) == 'N' ? 'T' : 'N', dA, A, ȳ, 1.0, b̄))
#     generate_primitive(f, [:(V <: StridedMatrix), :(W <: StridedVector)],
#         [:ul, :ta, :dA, :A, :b], [:nothing, :nothing, :nothing, :Ā, :b̄],
#         [:Char, :Char, :Char, :V, :W], [false, false, false, true, true], :y, :ȳ,
#         [:nothing, :nothing, :nothing, Ā, b̄], [:nothing, :nothing, :nothing, Ā!, b̄!])
# end

# # trsm
# function Ā_trsm(Y, Ȳ, side, ul, ta, dA, α, A, X)
#     Ā_full = uppercase(side) == 'L' ?
#         uppercase(ta) == 'N' ?
#             trsm('L', ul, 'T', dA, -1.0, A, Ȳ * Y.') :
#             trsm('R', ul, 'T', dA, -1.0, A, Y * Ȳ.') :
#         uppercase(ta) == 'N' ?
#             trsm('R', ul, 'T', dA, -1.0, A, Y.'Ȳ) :
#             trsm('L', ul, 'T', dA, -1.0, A, Ȳ.'Y)
#     return (uppercase(ul) == 'L' ? tril! : triu!)(Ā_full)
# end
# function X̄_trsm(Y, Ȳ, side, ul, ta, dA, α, A, X)
#     uppercase(side) == 'L' ?
#         uppercase(ta) == 'N' ?
#             trsm('L', ul, 'T', dA, α, A, Ȳ) :
#             trsm('L', ul, 'N', dA, α, A, Ȳ) :
#         uppercase(ta) == 'N' ?
#             trsm('R', ul, 'T', dA, α, A, Ȳ) :
#             trsm('R', ul, 'N', dA, α, A, Ȳ)
# end
# let f = :trsm
#     ᾱ, ᾱ! = :(ᾱ = sum(Ȳ .* Y) / α), :(ᾱ += sum(Ȳ .* Y) / α)
#     Ā = :(Ā = Ā_trsm(Y, Ȳ, side, ul, ta, dA, α, A, X))
#     Ā! = :(broadcast!((ā, δā)->ā + δā, Ā, Ā, Ā_trsm(Y, Ȳ, side, ul, ta, dA, α, A, X)))
#     X̄ = :(X̄ = X̄_trsm(Y, Ȳ, side, ul, ta, dA, α, A, X))
#     X̄! = :(broadcast!((x̄, δx̄)->x̄ + δx̄, X̄, X̄, X̄_trsm(Y, Ȳ, side, ul, ta, dA, α, A, X)))
#     generate_primitive(f, [:(T <: Number), :(V <: StridedMatrix), :(W <: StridedMatrix)],
#         [:side, :ul, :ta, :dA, :α, :A, :X],
#         [:nothing, :nothing, :nothing, :nothing, :ᾱ, :Ā, :X̄],
#         [:Char, :Char, :Char, :Char, :T, :V, :W],
#         [false, false, false, false, true, true, true], :Y, :Ȳ,
#         [:nothing, :nothing, :nothing, :nothing, ᾱ, Ā, X̄],
#         [:nothing, :nothing, :nothing, :nothing, ᾱ!, Ā!, X̄!])
# end

# # trsv
# let f = :trsv
#     Ā = :(Ā = Ā_trsm(y, ȳ, 'L', ul, ta, dA, 1.0, A, x))
#     Ā! = :(broadcast!((ā, δā)->ā, Ā, Ā, Ā_trsm(y, ȳ, 'L', ul, ta, dA, 1.0, A, x)))
#     x̄ = :(x̄ = trsv(ul, uppercase(ta) == 'N' ? 'T' : 'N', dA, A, ȳ))
#     x̄! = :(trsv!(ul, uppercase(ta) == 'N' ? 'T' : 'N', dA, A, ȳ, 1.0, x̄))
#     # @eval @primitive $f(ul::Char, ta::Char, dA::Char, A::V, x::W) y ȳ false false false $Ā $x̄
#     generate_primitive(f, [:(V <: StridedMatrix), :(W <: StridedVector)],
#         [:ul, :ta, :dA, :A, :x], [:nothing, :nothing, :nothing, :Ā, :x̄],
#         [:Char, :Char, :Char, :V, :W], [false, false, false, true, true], :y, :ȳ,
#         [:nothing, :nothing, :nothing, Ā, x̄], [:nothing, :nothing, :nothing, Ā!, x̄!])
# end

# # # TODO: Banded matrix operations.
# # # gbmv
# # # sbmv
