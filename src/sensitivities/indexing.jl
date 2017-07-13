export getindex

# Implementation of reverse-mode sensitivities for `getindex`.
eval(DiffBase, add_intercept(:getindex, :(Base.getindex), :(Tuple{Any, Vararg})))
@inline getindex(tape::Tape, node::Node) = Base.getindex(tape, node)
@inline function ∇(Ā, ::typeof(getindex), ::Type{Arg{1}}, p, y, ȳ, A, inds...)
    return Base.setindex!(Ā, ȳ, inds...)
end
@inline function ∇(::typeof(getindex), ::Type{Arg{1}}, p, y, ȳ, A, inds...)
    return ∇(zeros(A), getindex, Arg{1}, p, y, ȳ, A, inds...)
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
