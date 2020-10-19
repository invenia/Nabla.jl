# Hand code the identity because it's really fundamental. It doesn't need to generate a new
# node on the computational graph since it does nothing, but it is useful to have it's
# gradient implemented for use in higher-order functions.
import Base.identity
@explicit_intercepts identity Tuple{Any}
@inline ∇(::typeof(identity), ::Type{Arg{1}}, p, y, ȳ, x) = ȳ
@inline ∇(::typeof(identity), ::Type{Arg{1}}, x::Real) = one(x)

# Add method to resolve exponentiation ambiguity.
^(n::Node{<:Real}, p::Integer) = invoke(^, Tuple{Node{<:Real}, Real}, n, p)

import Base: float
@explicit_intercepts float Tuple{∇ArrayOrScalar}
∇(::typeof(float), ::Type{Arg{1}}, p, y, ȳ, x) = float(ȳ)
