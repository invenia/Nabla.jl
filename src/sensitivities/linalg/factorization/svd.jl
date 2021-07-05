import LinearAlgebra: svd
import Base: getproperty

# Iteration allows destructuring, e.g. U, S, V = svd(A)
# These definitions mirror those defined in the LinearAlgebra module, see
# https://github.com/JuliaLang/julia/blob/master/stdlib/LinearAlgebra/src/svd.jl#L20-L24
Base.iterate(usv::Branch{<:SVD}) = (usv.U, Val(:S))
Base.iterate(usv::Branch{<:SVD}, ::Val{:S}) = (usv.S, Val(:V))
Base.iterate(usv::Branch{<:SVD}, ::Val{:V}) = (usv.V, Val(:done))
Base.iterate(usv::Branch{<:SVD}, ::Val{:done}) = nothing
