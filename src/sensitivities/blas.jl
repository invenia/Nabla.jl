# import Base.LinAlg.BLAS: asum, dot, blascopy!, nrm2, scal, scal!, gemm, gemv, syrk, symm,
#     symv, trmm, trsm, trmv, trsv

# let f = :dot, par_types = [:(T <: StridedArray), :(V <: StridedArray)]

#     # Simple Julia dot.
#     new_x̄ = :(x̄ = z̄ * y)
#     update_x̄ = :(broadcast!((x̄, y, z̄)->x̄ + z̄ * y, x̄, x̄, y, z̄))
#     new_ȳ = :(ȳ = z̄ * x)
#     update_ȳ = :(broadcast!((x, ȳ, z̄)->ȳ + z̄ * x, ȳ, x, ȳ, z̄))
#     generate_primitive(f, par_types, [:x, :y], [:x̄, :ȳ], [:T, :V], [true, true], :z, :z̄,
#         [new_x̄, new_ȳ], [update_x̄, update_ȳ])

#     # Full BLAS dot.
#     new_x̄ = :(x̄ = scal!(n, z̄, blascopy!(n, y, iy, zeros(x), ix), ix))
#     update_x̄ = :(broadcast!((x̄, δx̄)->x̄ + δx̄, x̄, x̄, scal!(n, z̄, blascopy!(n, y, iy, zeros(x), ix), ix)))
#     new_ȳ = :(ȳ = scal!(n, z̄, blascopy!(n, x, ix, zeros(y), iy), iy))
#     update_ȳ = :(broadcast!((ȳ, δȳ->ȳ + δȳ, scal!(n, z̄, blascopy!(n, x, ix, zeros(y), iy), iy))))
#     generate_primitive(f, par_types, [:n, :x, :ix, :y, :iy], [:n̄, :x̄, :ix̄, :ȳ, :iȳ],
#         [:Int, :T, :Int, :V, :Int], [false, true, false, true, false], :z, :z̄,
#         [:nothing, new_x̄, :nothing, new_ȳ, :nothing],
#         [:nothing, update_x̄, :nothing, update_ȳ, :nothing])
# end

# let f = :nrm2, par_types = [:(T <: StridedArray)]

#     # Simple Julia nrm2.
#     new_x̄ = :(x̄ = x * (z̄ / z))
#     update_x̄ = :(broadcast!((x, c, x̄)->x̄ + c * x, x̄, x, z̄ / z, x̄))
#     generate_primitive(f, par_types, [:x], [:x̄], [:T], [true], :z, :z̄, [new_x̄], [update_x̄])

#     # Full BLAS nrm2.
#     new_x̄ = :(x̄ = scal!(n, z̄ / z, blascopy!(n, x, inc, zeros(x), inc), inc))
#     update_x̄ = :(broadcast!((x̄, δx̄)->x̄ + δx̄, scal!(n, z̄ / z, blascopy!(n, x, inc, zeros(x), inc), inc)))
#     generate_primitive(f, par_types, [:n, :x, :inc], [:nothing, :x̄, :nothing],
#         [:Int, :T, :Int], [false, true, false], :z, :z̄, [:nothing, new_x̄, :nothing],
#         [:nothing, update_x̄, :nothing])
# end

# let f = :asum, par_types = [:(T <: AbstractArray)]

#     # Simple Julia asum.
#     new_x̄ = :(x̄ = broadcast((x, z̄)->z̄ * sign(x), x, z̄))
#     update_x̄ = :(broadcast!((x̄, x, z̄)->x̄ + z̄ * sign(x), x̄, x, z̄))
#     generate_primitive(f, par_types, [:x], [:x̄], [:T], [true], :z, :z̄, [new_x̄], [update_x̄])

#     # Full BLAS asum.
#     new_x̄ = :(x̄ = scal!(n, z̄, blascopy!(n, sign(x), inc, zeros(x), inc), inc))
#     update_x̄ = :(broadcast!((x̄, δx̄)->x̄ + δx̄, scal!(n, z̄, blascopy!(n, sign(x), inc, zeros(x), inc), inc)))
#     generate_primitive(f, par_types, [:n, :x, :inc], [:nothing, :x̄, :nothing],
#         [:Int, :T, :Int], [false, true, false], :z, :z̄, [:nothing, new_x̄, :nothing],
#         [:nothing, update_x̄, :nothing])
# end

# # Some weird stuff going on that I haven't figured out yet.
# # let f = :(scal{T <: AbstractArray, V <: AbstractFloat})
# #     ā = :(blascopy!(n, z̄, inc, zeros(X), inc) .* X)
# #     X̄ = :(scal!(n, a, z̄, inc))
# #     @eva; @primitive $f(n::Int, a::V, X::T, inc::Int) z z̄ false $ā $X̄ false
# # end

# # gemm
# function Ā_gemm(Ȳ, tA, tB, α, A, B)
#     uppercase(tA) == 'N' ?
#         uppercase(tB) == 'N' ?
#             gemm('N', 'T', α, Ȳ, B) :
#             gemm('N', 'N', α, Ȳ, B) :
#         uppercase(tB) == 'N' ?
#             gemm('N', 'T', α, B, Ȳ) :
#             gemm('T', 'T', α, B, Ȳ)
# end
# function Ā!_gemm(Ȳ, tA, tB, α, A, B, Ā)
#     uppercase(tA) == 'N' ?
#         uppercase(tB) == 'N' ?
#             gemm!('N', 'T', α, Ȳ, B, 1.0, Ā) :
#             gemm!('N', 'N', α, Ȳ, B, 1.0, Ā) :
#         uppercase(tB) == 'N' ?
#             gemm!('N', 'T', α, B, Ȳ, 1.0, Ā) :
#             gemm!('T', 'T', α, B, Ȳ, 1.0, Ā)
# end
# function B̄_gemm(Ȳ, tA, tB, α, A, B)
#     uppercase(tA) == 'N' ?
#         uppercase(tB) == 'N' ?
#             gemm('T', 'N', α, A, Ȳ) :
#             gemm('T', 'N', α, Ȳ, A) :
#         uppercase(tB) == 'N' ?
#             gemm('N', 'N', α, A, Ȳ) :
#             gemm('T', 'T', α, Ȳ, A)
# end
# function B̄!_gemm(Ȳ, tA, tB, α, A, B, B̄)
#     uppercase(tA) == 'N' ?
#         uppercase(tB) == 'N' ?
#             gemm!('T', 'N', α, A, Ȳ, 1.0, B̄) :
#             gemm!('T', 'N', α, Ȳ, A, 1.0, B̄) :
#         uppercase(tB) == 'N' ?
#             gemm!('N', 'N', α, A, Ȳ, 1.0, B̄) :
#             gemm!('T', 'T', α, Ȳ, A, 1.0, B̄)
# end
# let f = :gemm
#     par_types = [:(T <: StridedMatrix), :(V <: StridedMatrix), :(W <: AbstractFloat)]
#     ᾱ, ᾱ! = :(ᾱ = sum(Ȳ .* Y) / α), :(ᾱ + sum(Ȳ .* Y) / α)
#     Ā, Ā! = :(Ā = Ā_gemm(Ȳ, tA, tB, α, A, B)), :(Ā!_gemm(Ȳ, tA, tB, α, A, B))
#     B̄, B̄! = :(B̄ = B̄_gemm(Ȳ, tA, tB, α, A, B)), :(B̄!_gemm(Ȳ, tA, tB, α, A, B))
#     generate_primitive(f, par_types, [:tA, :tB, :α, :A, :B],
#         [:nothing, :nothing, :ᾱ, :Ā, :B̄], [:Char, :Char, :W, :T, :V],
#         [false, false, true, true, true], :Y, :Ȳ,
#         [:nothing, :nothing, ᾱ, Ā, B̄], [:nothing, :nothing, ᾱ!, Ā!, B̄!])
# end

# let f = :gemv
#     ᾱ, ᾱ! = :(ᾱ = dot(ȳ, y) / α), :(ᾱ + dot(ȳ, y) / α)
#     Ā = :(Ā = uppercase(tA) == 'N' ? α * ȳ * x.' : α * x * ȳ.')
#     Ā! = :(uppercase(tA) == 'N' ? ger!(α, ȳ, x, Ā) : ger!(α, x, ȳ, Ā))
#     x̄ = :(x̄ = gemv(uppercase(tA) == 'N' ? 'T' : 'N', α, A, ȳ))
#     x̄! = :(gemv!(uppercase(tA) == 'N' ? 'T' : 'N', α, A, ȳ, 1.0, x̄))
#     generate_primitive(f,
#         [:(T <: StridedMatrix), :(V <: StridedVector), :(W <: AbstractFloat)],
#         [:tA, :α, :A, :x], [:nothing, :ᾱ, :Ā, :x̄], [:Char, :W, :T, :V],
#         [false, true, true, true], :y, :ȳ, [:nothing, ᾱ, Ā, x̄], [:nothing, ᾱ!, Ā!, x̄!])
# end

# # syrk
# function Ā_syrk(Ȳ, uplo, trans, α, A)
#     triȲ = uppercase(uplo) == 'L' ? tril(Ȳ) : triu(Ȳ)
#     out = gemm('N', trans, α, triȲ .+ triȲ.', A)
#     return uppercase(trans) == 'N' ? out : out.'
# end
# function Ā!_syrk(Ȳ, uplo, trans, α, A, Ā)
#     triȲ = uppercase(uplo) == 'L' ? tril(Ȳ) : triu(Ȳ)
#     out = gemm('N', trans, α, triȲ .+ triȲ.', A)
#     return broadcast!((ā, δā)->ā+δā, Ā, Ā, uppercase(trans) == 'N' ? out : out.')
# end
# let f = :syrk
#     ᾱ = :(g! = uppercase(uplo) == 'L' ? tril! : triu!; ᾱ = sum(g!(Ȳ .* Y)) / α)
#     ᾱ! = :(g! = uppercase(uplo) == 'L' ? tril! : trui!; ᾱ += sum(g!(Ȳ .* Y)) / α)
#     Ā = :(Ā = Ā_syrk(Ȳ, uplo, trans, α, A))
#     Ā! = :(Ā!_syrk(Ȳ, uplo, trans, α, A, Ā))
#     generate_primitive(f, [:(T <: Number), :(V <: StridedMatrix)],
#         [:uplo, :trans, :α, :A], [:nothing, :nothing, :ᾱ, :Ā], [:Char, :Char, :T, :V],
#         [false, false, true, true], :Y, :Ȳ, [:nothing, :nothing, ᾱ, Ā],
#         [:nothing, :nothing, ᾱ!, Ā!])
# end

# # symm
# function Ā_symm(Ȳ, side, ul, α, A, B)
#     tmp = uppercase(side) == 'L' ? Ȳ * B.' : B.'Ȳ
#     g! = uppercase(ul) == 'L' ? tril! : triu!
#     return α * g!(tmp + tmp' - Diagonal(tmp))
# end
# function Ā!_symm(Ȳ, side, ul, α, A, B, Ā)
#     tmp = uppercase(side) == 'L' ? Ȳ * B.' : B.'Ȳ
#     g! = uppercase(ul) == 'L' ? tril! : triu!
#     return broadcast!((ā, δā)->ā + δā, Ā, Ā, α * g!(tmp + tmp' - Diagonal(tmp)))
# end
# let f = :symm
#     ᾱ, ᾱ! = :(ᾱ = sum(Ȳ .* Y) / α), :(ᾱ += sum(Ȳ .* Y) / α)
#     Ā, Ā! = :(Ā = Ā_symm(Ȳ, side, ul, α, A, B)), :(Ā!_symm(Ȳ, side, ul, α, A, B, Ā))
#     B̄, B̄! = :(B̄ = symm(side, ul, α, A, Ȳ)), :(symm!(side, ul, α, A, Ȳ, 1.0, B̄))
#     generate_primitive(f, [:(T <: Number), :(V <: StridedMatrix), :(W <: StridedMatrix)],
#         [:side, :ul, :α, :A, :B], [:nothing, :nothing, :ᾱ, :Ā, :B̄],
#         [:Char, :Char, :T, :V, :W], [false, false, true, true, true], :Y, :Ȳ,
#         [:nothing, :nothing, ᾱ, Ā, B̄], [:nothing, :nothing, ᾱ!, Ā!, B̄!])
# end

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
