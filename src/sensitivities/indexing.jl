# Implementation of reverse-mode sensitivities for `getindex`.
@primitive getindex(args...) where {__CONTEXT__ <: ∇Ctx} = propagate_forward(getindex, args...)

function ∇(Ā, ::typeof(getindex), ::Type{Val{1}}, p, y, ȳ, A, inds...)
    Ā[inds...] .+= ȳ
    return Ā
end
function ∇(Ā, ::typeof(getindex), ::Type{Val{1}}, p, y::AbstractArray, ȳ::AbstractArray, A, inds...)
    Ā[inds...] .+= reshape(ȳ, size(y)...)
    return Ā
end
function ∇(::typeof(getindex), ::Type{Val{1}}, p, y, ȳ, A, inds...)
    return ∇(zerod_container(A), getindex, Val{1}, p, y, ȳ, A, inds...)
end

# # Implementation of reverse-mode sensitivities for `view`. Not currently in use because
# `view` turns out to actually be a bit awkward.
# eval(DiffBase, add_intercept(:view, :(Base.view), :(Tuple{Any, Vararg})))
# @inline function ∇(Ā, ::typeof(view), ::Type{Val{1}}, p, y, ȳ, A, inds...)
#     return Base.setindex!(Ā, ȳ, inds...)
# end
# @inline function ∇(::typeof(view), ::Type{Val{1}}, p, y, ȳ, A, inds...)
#     return ∇(zeros(A), view, Val{1}, p, y, ȳ, A, inds...)
# end
