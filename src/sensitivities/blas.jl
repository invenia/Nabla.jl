import Base.LinAlg.BLAS: asum, dot, blascopy!, nrm2, scal, scal!

# Dot product.
function x̄(z̄, n::Int, x::AbstractArray, incx, y::AbstractArray, incy)
    return scal!(n, z̄, blascopy!(n, y, incy, zeros(x), incx), incx)
end
@primitive dot{T, V <: AbstractArray}(x::T, y::V) z z̄ z̄ * y z̄ * x
@primitive dot{T, V <: AbstractArray}(n::Int, x::T, ix::Int, y::V, iy::Int) z z̄ false x̄(z̄, n, x, ix, y, iy) false x̄(z̄, n, y, iy, x, ix) false

# nrm2
@primitive nrm2{T <: AbstractArray}(x::T) z z̄ x * (z̄ / z)
@primitive nrm2{T <: AbstractArray}(n::Int, x::T, inc::Int) z z̄ false scal!(n, z̄ / z, blascopy!(n, x, inc, zeros(x), inc), inc) false

# asum
@primitive asum{T <: AbstractArray}(x::T) z z̄ z̄ * sign(x)
@primitive asum{T <: AbstractArray}(n::Int, x::T, inc::Int) z z̄ false scal!(n, z̄, blascopy!(n, sign(x), inc, zeros(x), inc), inc) false

# # scal
# @primitive scal{T <: AbstractArray, V <: AbstractFloat}(n::Int, a::V, X::T, inc::Int) z z̄ false blascopy!(n, z̄, inc, zeros(X), inc) .* X scal!(n, a, z̄, inc) false

# # gemm
# function gemmᾱ{T, V <: AbstractArray, W <: AbstractFloat}(Ȳ, Y, tA::Char, tB::Char, α::W, A::T, B::V)
#     return Ȳ .* Y
# end
# function gemmĀ{T, V <: AbstractArray, W <: AbstractFloat}(tA::Char, tB::Char, α::W, A::T, B::V)
#     if (tA == 'N' || tA == 'n') && (tB == 'N' || tB == 'n')
#         return 
#     elseif

#     elseif

#     elseif
#         throw(ArgumentError("Invalid tA or tB."))
#     end
# end
# function gemmB̄{T, V <: AbstractArray, W <: AbstractFloat}(tA::Char, tB::Char, α::W, A::T, B::V)
#     if (tA == 'N' || tA == 'n') && (tB == 'N' || tB == 'n')
#         return 
#     elseif

#     elseif

#     elseif
#         throw(ArgumentError("Invalid tA or tB."))
#     end
# end
# @primitive gemm{T, V <: AbstractArray, W <: AbstractFloat}(tA::Char, tB::Char, α::W, A::T, B::V) Y Ȳ false false Ȳ .* Y gemmĀ(tA, tB, α, A, B) gemmB̄(tA, tB, α, A, B)

# syrk
# herk
# gbmv
# sbmv
# gemm
# gemv
# symm
# symv
# trmm
# trsm
# trmv
