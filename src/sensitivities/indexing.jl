export getindex

# A collection of methods for initialising containers to zero.
@noinline zerod_container(x::Number) = zero(x)
@noinline zerod_container(x::AbstractArray{T} where T<:Real) = zeros(x)
@noinline zerod_container(x::Tuple) = ([zerod_container(n) for n in x])
@noinline function zerod_container(x)
    y = Base.copy(x)
    [Base.setindex!(y, zerod_container(y[n]), n) for n in eachindex(y)]
    return y
end

# Implementation of reverse-mode sensitivities for `getindex`.
eval(DiffBase, add_intercept(:getindex, :(Base.getindex), :(Tuple{Any, Vararg})))
@inline getindex(tape::Tape, node::Node) = Base.getindex(tape, node)
@inline function ∇(Ā, ::typeof(getindex), ::Type{Arg{1}}, p, y, ȳ, A, inds...)
    return Base.setindex!(Ā, ȳ, inds...)
end
@inline function ∇(::typeof(getindex), ::Type{Arg{1}}, p, y, ȳ, A, inds...)
    return ∇(zerod_container(A), getindex, Arg{1}, p, y, ȳ, A, inds...)
end

# # Implementation of reverse-mode sensitivities for `view`. Not currently in use because
# `view` turns out to actually be a bit awkward.
# eval(DiffBase, add_intercept(:view, :(Base.view), :(Tuple{Any, Vararg})))
# @inline function ∇(Ā, ::typeof(view), ::Type{Arg{1}}, p, y, ȳ, A, inds...)
#     return Base.setindex!(Ā, ȳ, inds...)
# end
# @inline function ∇(::typeof(view), ::Type{Arg{1}}, p, y, ȳ, A, inds...)
#     return ∇(zeros(A), view, Arg{1}, p, y, ȳ, A, inds...)
# end
