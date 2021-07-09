import LinearAlgebra: qr
import Base: getproperty

const QRLike = Union{QR, LinearAlgebra.QRCompactWY}

@explicit_intercepts qr Tuple{AbstractMatrix{<:Real}}

function ∇(
    ::typeof(qr),
    ::Type{Arg{1}},
    p,
    Y::QRLike,
    Ȳ::NamedTuple{(:Q,:R)},
    A::AbstractMatrix,
)
    Q, R = Y
    Q̄, R̄ = Ȳ
    triu!(R̄)
    M = R*R̄'
    M .-= Q̄'Q
    return (Q̄ + Q*Symmetric(M, :L)) / R'
end

@explicit_intercepts getproperty Tuple{QRLike, Symbol} [true, false]

function ∇(::typeof(getproperty), ::Type{Arg{1}}, p, y, ȳ, F::QRLike, x::Symbol)
    if x === :Q
        return (Q=reshape(ȳ, size(F.Q)), R=zeroslike(F.R))
    elseif x === :R
        return (Q=zeroslike(F.Q), R=reshape(ȳ, size(F.R)))
    else
        throw(ArgumentError("unrecognized property $x; expected Q or R"))
    end
end

function ∇(
    x̄::NamedTuple{(:Q,:R)},
    ::typeof(getproperty),
    ::Type{Arg{1}},
    p, y, ȳ,
    F::QRLike,
    x::Symbol,
)
    x̄_update = ∇(getproperty, Arg{1}, p, y, ȳ, F, x)
    if x === :Q
        return (Q=update!(x̄.Q, x̄_update.Q), R=x̄.R)
    elseif x === :R
        return (Q=x̄.Q, R=update!(x̄.R, x̄_update.R))
    end
end

Base.iterate(qr::Branch{<:QRLike}) = (qr.Q, Val(:R))
Base.iterate(qr::Branch{<:QRLike}, ::Val{:R}) = (qr.R, Val(:done))
Base.iterate(qr::Branch{<:QRLike}, ::Val{:done}) = nothing
