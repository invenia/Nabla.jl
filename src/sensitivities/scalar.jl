# Add method to resolve exponentiation ambiguity.
^(n::Node{<:Real}, p::Integer) = invoke(^, Tuple{Node{<:Real}, Real}, n, p)

import Base: float
@explicit_intercepts float Tuple{∇ArrayOrScalar}
∇(::typeof(float), ::Type{Arg{1}}, p, y, ȳ, x) = float(ȳ)
