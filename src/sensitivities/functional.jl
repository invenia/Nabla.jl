import Base: map, broadcast, mapreduce, mapreducedim, size, RefValue
import Base.Broadcast: broadcast_shape

# Some horrible type piracy that is necessary for the change in broadcast / map semantics.
size(x::RefValue{<:Number}) = (1,)

# Implementation of sensitivities w.r.t. `map`.
@∇primitive Base.map
has∇definition(::typeof(map), ::Any, ::Vararg{∇Array}) = true

# Compute sensitivity w.r.t. the N^{th} input, N > 1.
∇(::typeof(map), ::Type{Val{N}}, p, y, ȳ, f, A::∇Array...) where N =
    _∇(map, Val{N-1}, p, y, ȳ, f, A...)
_∇(::typeof(map), arg::Type{Val{N}}, p, y, ȳ, f, A::∇Array...) where N =
    hasmethod(∇, Tuple{typeof(f), Type{Val{N}}, Any, Any, Any, map(eltype, A)...}) ?
        map((yn, ȳn, An...)->∇(f, Val{N}, p, yn, ȳn, An...), y, ȳ, A...) :
        map((ȳn, An...)->ȳn * fmad(f, An, Val{N}), ȳ, A...)

# Implementation of sensitivities w .r.t. `broadcast`.
@∇primitive Base.broadcast

"""
    broadcastsum!(f::Function, add::Bool, z, As...)

Broadcast f over As and reduce to z by summing. If add is true, then the result is added to
the current value of z, otherwise it is overwritten.
"""
function broadcastsum!(f::Function, add::Bool, z, As...)
    tmp_shape = broadcast_shape(map(size, As)...)
    size(z) != tmp_shape ?
        sum!(z, broadcast!(f, Array{eltype(z)}(undef, tmp_shape), As...), init=!add) :
        add ? broadcast!((z, x...)->z + f(x...), z, z, As...) : broadcast!(f, z, As...)
end

"""
    broadcastsum(f::Function, add::Bool, z::AbstractArray, As...)

Allocating version of broadcastsum! specialised for Arrays.
"""
broadcastsum(f::Function, add::Bool, z::AbstractArray, As...) =
    broadcastsum!(f, add, Array{eltype(z)}(undef, size(z)), As...)

"""
    broadcastsum(f::Function, add::Bool, z::Number, As...)

Specialisation of broadcastsum to Number-sized outputs.
"""
function broadcastsum(f::Function, add::Bool, z::Number, As...)
    tmp = Array{eltype(z)}(undef, broadcast_shape(map(size, As)...))
    return sum(broadcast!(f, tmp, As...)) + (add ? z : zero(z))
end

# Compute sensitivity w.r.t. the N^{th} input, N > 1.
∇(::typeof(broadcast), ::Type{Val{N}}, p, y, ȳ, f::Function, A::∇ArrayOrScalar...) where N =
    _∇(broadcast, Val{N-1}, p, y, ȳ, f, A...)
_∇(::typeof(broadcast), ::Type{Val{N}}, p, y, ȳ, f, A...) where N =
    hasmethod(∇, Tuple{typeof(f), Type{Val{N}}, Any, Any, Any, map(eltype, A)...}) ?
        broadcastsum((yn, ȳn, xn...)->∇(f, Val{N}, p, yn, ȳn, xn...), false, A[N], y, ȳ, A...) :
        broadcastsum((ȳn, xn...)->ȳn * fmad(f, xn, Val{N}), false, A[N], ȳ, A...)

@∇primitive mapreduce
has∇definition(::typeof(mapreduce), ::Any, ::typeof(Base.add_sum), ::∇Array) = true
has∇definition(::typeof(mapreduce), ::Any, ::typeof(+), ::∇Array) = true
∇(::typeof(mapreduce), ::Type{Val{3}}, p, y, ȳ, f, ::typeof(Base.add_sum), A::∇ArrayOrScalar) =
    hasmethod(∇, Tuple{typeof(f), Type{Val{1}}, Real}) ?
        broadcast(An->ȳ * ∇(f, Val{1}, An), A) :
        broadcast(An->ȳ * fmad(f, (An,), Val{1}), A)
∇(::typeof(mapreduce), ::Type{Val{3}}, p, y, ȳ, f, ::typeof(+), A::∇ArrayOrScalar) =
    hasmethod(∇, Tuple{typeof(f), Type{Val{1}}, Real}) ?
        broadcast(An->ȳ * ∇(f, Val{1}, An), A) :
        broadcast(An->ȳ * fmad(f, (An,), Val{1}), A)

@∇primitive mapreducedim
has∇definition(::typeof(mapreducedim), ::Any, ::typeof(Base.add_sum), ::∇Array, ::Int, ::Any) = true
has∇definition(::typeof(mapreducedim), ::Any, ::typeof(+), ::∇Array, ::Int, ::Any) = true
∇(::typeof(mapreducedim),
    ::Type{Val{3}},
    p, y, ȳ, f,
    ::typeof(+),
    A::AbstractArray{<:∇Scalar},
    region,
) = hasmethod(∇, Tuple{typeof(f), Type{Val{1}}, ∇Scalar}) ?
        broadcast((An, ȳn)->ȳn * ∇(f, Val{1}, An), A, ȳ) :
        broadcast((An, ȳn)->ȳn * fmad(f, (An,), Val{1}), A, ȳ)



# # Intercepts for `mapreduce`, `mapfoldl` and `mapfoldr` under `op` `+`.
# type_tuple = :(Tuple{Any, typeof(+), ∇ArrayOrScalar})
# for f in (:mapreduce, :mapfoldl, :mapfoldr)

#     @eval @primitive $f(x...) where __CONTEXT__ <: ∇Ctx = propagate_forward($f, x...)
#     @eval ∇(::typeof($f), ::Type{Val{3}}, p, y, ȳ, f, ::typeof(Base.add_sum), A::∇ArrayOrScalar) =
#         hasmethod(∇, Tuple{typeof(f), Type{Val{1}}, Real}) ?
#             broadcast(An->ȳ * ∇(f, Val{1}, An), A) :
#             broadcast(An->ȳ * fmad(f, (An,), Val{1}), A)
#     @eval ∇(::typeof($f), ::Type{Val{3}}, p, y, ȳ, f, ::typeof(+), A::∇ArrayOrScalar) =
#         hasmethod(∇, Tuple{typeof(f), Type{Val{1}}, Real}) ?
#             broadcast(An->ȳ * ∇(f, Val{1}, An), A) :
#             broadcast(An->ȳ * fmad(f, (An,), Val{1}), A)
# end

# @∇primitive Base.mapreduce


# accept_wo_default = :(Tuple{Function, typeof(+), AbstractArray{<:∇Scalar}, Any})
# accept_w_default = :(Tuple{Function, typeof(+), AbstractArray{<:∇Scalar}, Any, ∇Scalar})
# @primtive mapreducedim(x...) where __CONTEXT__ <: ∇Ctx = propagate_forward(mapreducedim, x...)

# ∇(::typeof(mapreducedim),
#     ::Type{Val{3}},
#     p, y, ȳ, f,
#     ::typeof(+),
#     A::AbstractArray{<:∇Scalar},
#     region,
#     v0=nothing,
# ) = method_exists(∇, Tuple{typeof(f), Type{Val{1}}, ∇Scalar}) ?
#         broadcast((An, ȳn)->ȳn * ∇(f, Val{1}, An), A, ȳ) :
#         broadcast((An, ȳn)->ȳn * fmad(f, (An,), Val{1}), A, ȳ)

# # Make `sum` work. It currently fails as the type specification is too restrictive.
# sum(n::Node{<:AbstractArray}, region) = mapreducedim(identity, +, n, region)
