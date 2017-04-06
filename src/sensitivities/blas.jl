import Base.LinAlg.BLAS: asum, dot, blascopy!, nrm2, scal, scal!, gemm, gemv, syrk, symm,
    symv, trmm, trsm, trmv, trsv

let f = :(dot{T, V <: AbstractArray})

    # Simple Julia dot.
    x̄ = :(z̄ * y)
    ȳ = :(z̄ * x)
    @eval @primitive $f(x::T, y::V) z z̄ $x̄ $ȳ

    # Full BLAS dot.
    x̄ = :(scal!(n, z̄, blascopy!(n, y, iy, zeros(x), ix), ix))
    ȳ = :(scal!(n, z̄, blascopy!(n, x, ix, zeros(y), iy), iy))
    @eval @primitive $f(n::Int, x::T, ix::Int, y::V, iy::Int) z z̄ false $x̄ false $ȳ false
end

let f = :(nrm2{T <: AbstractArray})

    # Simple Julia nrm2.
    x̄ = :(x * (z̄ / z))
    @eval @primitive $f(x::T) z z̄ $x̄

    # Full BLAS nrm2.
    x̄ = :(scal!(n, z̄ / z, blascopy!(n, x, inc, zeros(x), inc), inc))
    @eval @primitive $f(n::Int, x::T, inc::Int) z z̄ false $x̄ false
end

let f = :(asum{T <: AbstractArray})

    # Simple Julia asum.
    x̄ = :(z̄ * sign(x))
    @eval @primitive $f(x::T) z z̄ $x̄

    # Full BLAS asum.
    x̄ = :(scal!(n, z̄, blascopy!(n, sign(x), inc, zeros(x), inc), inc))
    @eval @primitive $f(n::Int, x::T, inc::Int) z z̄ false $x̄ false
end

# Some weird stuff going on that I haven't figured out yet.
# let f = :(scal{T <: AbstractArray, V <: AbstractFloat})
#     ā = :(blascopy!(n, z̄, inc, zeros(X), inc) .* X)
#     X̄ = :(scal!(n, a, z̄, inc))
#     @eva; @primitive $f(n::Int, a::V, X::T, inc::Int) z z̄ false $ā $X̄ false
# end

# gemm
function Ā_gemm(Ȳ, tA, tB, α, A, B)
    uppercase(tA) == 'N' ?
        uppercase(tB) == 'N' ?
            gemm('N', 'T', α, Ȳ, B) :
            gemm('N', 'N', α, Ȳ, B) :
        uppercase(tB) == 'N' ?
            gemm('N', 'T', α, B, Ȳ) :
            gemm('T', 'T', α, B, Ȳ)
end
function B̄_gemm(Ȳ, tA, tB, α, A, B)
    uppercase(tA) == 'N' ?
        uppercase(tB) == 'N' ?
            gemm('T', 'N', α, A, Ȳ) :
            gemm('T', 'N', α, Ȳ, A) :
        uppercase(tB) == 'N' ?
            gemm('N', 'N', α, A, Ȳ) :
            gemm('T', 'T', α, Ȳ, A)
end
let f = :(gemm{T, V <: AbstractMatrix, W <: AbstractFloat})
    ᾱ = :(sum(Ȳ .* Y / α))
    Ā = :(Ā_gemm(Ȳ, tA, tB, α, A, B))
    B̄ = :(B̄_gemm(Ȳ, tA, tB, α, A, B))
    @eval @primitive $f(tA::Char, tB::Char, α::W, A::T, B::V) Y Ȳ false false $ᾱ $Ā $B̄
end

# gemv
let f = :(gemv{T <: StridedMatrix, V <: StridedVector, W <: AbstractFloat})
    ᾱ = :(dot(ȳ, y) / α)
    Ā = :(uppercase(tA) == 'N' ? α * ȳ * x.' : α * x * ȳ.')
    B̄ = :(uppercase(tA) == 'N' ? gemv('T', α, A, ȳ) : gemv('N', α, A, ȳ))
    @eval @primitive $f(tA::Char, α::W, A::T, x::V) y ȳ false $ᾱ $Ā $B̄
end

# syrk
function Ā_syrk(Ȳ, uplo, trans, α, A)
    triȲ = uppercase(uplo) == 'L' ? tril(Ȳ) : triu(Ȳ)
    out = gemm('N', trans, α, triȲ .+ triȲ.', A)
    return uppercase(trans) == 'N' ? out : out.'
end
let f = :(syrk{T <: Number, V <: StridedMatrix})
    ᾱ = :(g! = uppercase(uplo) == 'L' ? tril! : triu!; sum(g!(Ȳ .* Y)) / α)
    Ā = :(Ā_syrk(Ȳ, uplo, trans, α, A))
    @eval @primitive $f(uplo::Char, trans::Char, α::T, A::V) Y Ȳ false false $ᾱ $Ā
end

# symm
function Ā_symm(Ȳ, side, ul, α, A, B)
    tmp = uppercase(side) == 'L' ? Ȳ * B.' : B.'Ȳ
    g! = uppercase(ul) == 'L' ? tril! : triu!
    return α * g!(tmp + tmp' - Diagonal(tmp))
end
let f = :(symm{T <: Number, V <: StridedMatrix, W <: StridedMatrix})
    ᾱ = :(sum(Ȳ .* Y) / α)
    Ā = :(Ā_symm(Ȳ, side, ul, α, A, B))
    B̄ = :(symm(side, ul, α, A, Ȳ))
    @eval @primitive $f(side::Char, ul::Char, α::T, A::V, B::W) Y Ȳ false false $ᾱ $Ā $B̄
end

# symv
let f = :(symv{T <: Union{Float32, Float64}, V <: StridedMatrix, W <: StridedVector})
    ᾱ = :(dot(ȳ, y) / α)
    Ā = :(Ā_symm(ȳ, 'L', ul, α, A, x))
    x̄ = :(symv(ul, α, A, ȳ))
    @eval @primitive $f(ul::Char, α::T, A::V, x::W) y ȳ false $ᾱ $Ā $x̄
end

# trmm
function Ā_trmm(Y, Ȳ, side, ul, ta, dA, α, A, B)
    Ā_full = uppercase(side) == 'L' ?
        uppercase(ta) == 'N' ?
            gemm('N', 'T', α, Ȳ, B) :
            gemm('N', 'T', α, B, Ȳ) :
        uppercase(ta) == 'N' ?
            gemm('T', 'N', α, B, Ȳ) :
            gemm('T', 'N', α, Ȳ, B)
    return (uppercase(ul) == 'L' ? tril! : triu!)(Ā_full)
end
function B̄_trmm(Y, Ȳ, side, ul, ta, dA, α, A, B)
    uppercase(side) == 'L' ?
        uppercase(ta) == 'N' ?
            trmm('L', ul, 'T', dA, α, A, Ȳ) :
            trmm('L', ul, 'N', dA, α, A, Ȳ) :
        uppercase(ta) == 'N' ?
            trmm('R', ul, 'T', dA, α, A, Ȳ) :
            trmm('R', ul, 'N', dA, α, A, Ȳ)
end
let f = :(trmm{T <: Number, V <: StridedMatrix, W <: StridedMatrix})
    ᾱ = :(sum(Ȳ .* Y) / α)
    Ā = :(Ā_trmm(Y, Ȳ, side, ul, ta, dA, α, A, B))
    B̄ = :(B̄_trmm(Y, Ȳ, side, ul, ta, dA, α, A, B))
    @eval @primitive $f(side::Char, ul::Char, ta::Char, dA::Char, α::T, A::V, B::W) Y Ȳ false false false false $ᾱ $Ā $B̄
end

# trsv
let f = :(trmv{V <: StridedMatrix, W <: StridedVector})
    Ā = :((uppercase(ul) == 'L' ? tril! : triu!)(uppercase(ta) == 'N' ? ȳ * b.' : b * ȳ.'))
    b̄ = :(uppercase(ta) == 'N' ? trmv(ul, 'T', dA, A, ȳ) : trmv(ul, 'N', dA, A, ȳ))
    @eval @primitive $f(ul::Char, ta::Char, dA::Char, A::V, b::W) y ȳ false false false $Ā $b̄
end

# trsm
function Ā_trsm(Y, Ȳ, side, ul, ta, dA, α, A, X)
    Ā_full = uppercase(side) == 'L' ?
        uppercase(ta) == 'N' ?
            trsm('L', ul, 'T', dA, -1.0, A, Ȳ * Y.') :
            trsm('R', ul, 'T', dA, -1.0, A, Y * Ȳ.') :
        uppercase(ta) == 'N' ?
            trsm('R', ul, 'T', dA, -1.0, A, Y.'Ȳ) :
            trsm('L', ul, 'T', dA, -1.0, A, Ȳ.'Y)
    return (uppercase(ul) == 'L' ? tril! : triu!)(Ā_full)
end
function X̄_trsm(Y, Ȳ, side, ul, ta, dA, α, A, X)
    uppercase(side) == 'L' ?
        uppercase(ta) == 'N' ?
            trsm('L', ul, 'T', dA, α, A, Ȳ) :
            trsm('L', ul, 'N', dA, α, A, Ȳ) :
        uppercase(ta) == 'N' ?
            trsm('R', ul, 'T', dA, α, A, Ȳ) :
            trsm('R', ul, 'N', dA, α, A, Ȳ)
end
let f = :(trsm{T <: Number, V <: StridedMatrix, W <: StridedMatrix})
    ᾱ = :(sum(Ȳ .* Y) / α)
    Ā = :(Ā_trsm(Y, Ȳ, side, ul, ta, dA, α, A, X))
    X̄ = :(X̄_trsm(Y, Ȳ, side, ul, ta, dA, α, A, X))
    @eval @primitive $f(side::Char, ul::Char, ta::Char, dA::Char, α::T, A::V, X::W) Y Ȳ false false false false $ᾱ $Ā $X̄
end

# trsv
let f = :(trsv{V <: StridedMatrix, W <: StridedVector})
    Ā = :(Ā_trsm(y, ȳ, 'L', ul, ta, dA, 1.0, A, x))
    x̄ = :(uppercase(ta) == 'N' ? trsv(ul, 'T', dA, A, ȳ) : trsv(ul, 'N', dA, A, ȳ))
    @eval @primitive $f(ul::Char, ta::Char, dA::Char, A::V, x::W) y ȳ false false false $Ā $x̄
end

# TODO: Banded matrix operations.
# gbmv
# sbmv
