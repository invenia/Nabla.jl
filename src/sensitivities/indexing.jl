# Implementation of reverse-mode sensitivities for `getindex`.
import Base.getindex
for i = 1:7
    T = Expr(:curly, :Tuple, fill(:Any, i)...)
    is_node = Expr(:vect, true, fill(false, i - 1)...)
    @eval @explicit_intercepts getindex $T $is_node
end

function ∇(::typeof(getindex), ::Type{Arg{1}}, p, y, ȳ, A::Ref)
    return Ref(ȳ)
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
