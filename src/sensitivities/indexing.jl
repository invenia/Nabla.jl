# Implementation of reverse-mode sensitivities for `getindex`.
import Base.getindex
@explicit_intercepts getindex Tuple{Any, Any} [true, false]
@explicit_intercepts getindex Tuple{Any, Any, Any} [true, false, false]
@explicit_intercepts getindex Tuple{Any, Any, Any, Any} [true, false, false, false]
@explicit_intercepts getindex Tuple{Any, Any, Any, Any, Any} [true, false, false, false, false]
@explicit_intercepts getindex Tuple{Any, Any, Any, Any, Any, Any} [true, false, false, false, false, false]
@explicit_intercepts getindex Tuple{Any, Any, Any, Any, Any, Any, Any} [true, false, false, false, false, false, false]

∇(Ā, ::typeof(getindex), ::Type{Arg{1}}, p, y, ȳ, A, inds...) = setindex!(Ā, ȳ, inds...)
function ∇(::typeof(getindex), ::Type{Arg{1}}, p, y, ȳ, A, inds...)
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
