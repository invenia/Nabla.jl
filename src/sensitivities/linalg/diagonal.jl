import LinearAlgebra: det, logdet, diagm, Diagonal, diag

const ∇ScalarDiag = Diagonal{<:∇Scalar}

@explicit_intercepts Diagonal Tuple{∇AbstractMatrix}
function ∇(
    ::Type{Diagonal},
    ::Type{Arg{1}},
    p,
    Y::∇ScalarDiag,
    Ȳ::∇ScalarDiag,
    X::∇AbstractMatrix,
)
    X̄ = zeroslike(X)
    copyto!(view(X̄, diagind(X)), Ȳ.diag)
    return X̄
end
function ∇(
    X̄::∇AbstractMatrix,
    ::Type{Diagonal},
    ::Type{Arg{1}},
    p,
    Y::∇ScalarDiag,
    Ȳ::∇ScalarDiag,
    X::∇AbstractMatrix,
)
    X̄_diag = view(X̄, diagind(X̄))
    broadcast!(+, X̄_diag, X̄_diag, Ȳ.diag)
    return X̄
end


# NOTE: diagm can't go through the @explicit_intercepts machinery directly because as of
# Julia 0.7, its methods are not sufficiently straightforward; we need to dispatch on one
# of the parameters in the parametric type of diagm's one argument. However, we can cheat
# a little bit and use an internal helper function _diagm that has simple methods that
# dispatch to diagm when no arguments are Nodes, and we'll extend diagm to dispatch to
# _diagm when it receives arguments that are nodes. _diagm can go through the intercepts
# machinery, so it knows how to deal.

# TODO: Possibly we should overload `Pair` so that it constructs a `Node{Pair}` then this
# would hit sensitivities that we have defined via ChainRules.

_diagm(x::∇AbstractVector, k::Integer=0) = diagm(k => x)
LinearAlgebra.diagm(x::Pair{<:Integer, <:Node{<:∇AbstractVector}}) = _diagm(last(x), first(x))


@explicit_intercepts _diagm Tuple{∇AbstractVector}
function ∇(
    ::typeof(_diagm),
    ::Type{Arg{1}},
    p,
    Y::∇AbstractMatrix,
    Ȳ::∇AbstractMatrix,
    x::∇AbstractVector,
)
    return copyto!(similar(x), view(Ȳ, diagind(Ȳ)))
end
function ∇(
    x̄::∇AbstractVector,
    ::typeof(_diagm),
    ::Type{Arg{1}},
    p,
    Y::∇AbstractMatrix,
    Ȳ::∇AbstractMatrix,
    x::∇AbstractVector,
)
    return broadcast!(+, x̄, x̄, view(Ȳ, diagind(Ȳ)))
end
@explicit_intercepts _diagm Tuple{∇AbstractVector, Integer} [true, false]
function ∇(
    ::typeof(_diagm),
    ::Type{Arg{1}},
    p,
    Y::∇AbstractMatrix,
    Ȳ::∇AbstractMatrix,
    x::∇AbstractVector,
    k::Integer,
)
    return copyto!(similar(x), view(Ȳ, diagind(Ȳ, k)))
end
function ∇(
    x̄::∇AbstractVector,
    ::typeof(_diagm),
    ::Type{Arg{1}},
    p,
    Y::∇AbstractMatrix,
    Ȳ::∇AbstractMatrix,
    x::∇AbstractVector,
    k::Integer,
)
    return broadcast!(+, x̄, x̄, view(Ȳ, diagind(Ȳ, k)))
end
