# We ignore some rules from chainrules, here we reimplement them

# We ignore purely varadic, and we do not generate  overloads for such  methods

import Base: +
@eval @explicit_intercepts $(Symbol("+")) Tuple{∇Array, ∇Array}
@inline ∇(::typeof(+), ::Type{Arg{1}}, p, z, z̄, x::∇Array, y::∇Array) =
    ∇(broadcast, Arg{2}, p, z, z̄, +, x, y)
@inline ∇(::typeof(+), ::Type{Arg{2}}, p, z, z̄, x::∇Array, y::∇Array) =
    ∇(broadcast, Arg{3}, p, z, z̄, +, x, y)

import Base: vcat, hcat
@union_intercepts hcat Tuple{Vararg{∇Array}} Tuple{Vararg{AbstractArray}}
function Nabla.∇(
    ::typeof(hcat),
    ::Type{Arg{i}},
    _,
    y,
    ȳ,
    A::AbstractArray...
) where i
    l = sum([size(A[j], 2) for j in 1:(i - 1)])
    u = l + size(A[i], 2)
    # Using copy materializes the views returned by selectdim
    return copy(u > l + 1 ? selectdim(ȳ, 2, (l+1):u) : selectdim(ȳ, 2, u))
end

@union_intercepts vcat Tuple{Vararg{∇Array}} Tuple{Vararg{AbstractArray}}
function Nabla.∇(
    ::typeof(vcat),
    ::Type{Arg{i}},
    _,
    y,
    ȳ,
    A::AbstractArray...
) where i
    l = sum([size(A[j], 1) for j in 1:(i - 1)])
    u = l + size(A[i], 1)
    return copy(selectdim(ȳ, 1, (l+1):u))
end
