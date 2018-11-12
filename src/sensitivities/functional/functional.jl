# Implementation of sensitivities w.r.t. `map`.
import Base: map, BroadcastStyle

@generated function is_atom(ctx::∇Ctx, ::typeof(map), ::Any, A::∇MaybeTagged{<:∇Array}...)
    return any(a->istaggedtype(a, Ref(ctx)), A)
end

# Compute sensitivity w.r.t. the N^{th} input, N > 1.
function ∇(::typeof(map), ::Type{Arg{N}}, p, y, ȳ, f::Function, A::∇Array...) where N
    return _∇(map, Arg{N-1}, p, y, ȳ, f, A...)
end
function _∇(::typeof(map), arg::Type{Arg{N}}, p, y, ȳ, f::Function, A::∇Array...) where N
    if hasmethod(∇, Tuple{typeof(f), Type{Arg{N}}, Any, Any, Any, map(eltype, A)...})
        return map((yn, ȳn, An...)->∇(f, Arg{N}, p, yn, ȳn, An...), y, ȳ, A...)
    else
        return map((ȳn, An...)->ȳn * fmad(f, An, Val{N}), ȳ, A...)
    end
end

# # Deal with ambiguities introduced by `map`.
# map(f, x::AbstractArray{<:Number}...) = invoke(map, Tuple{Any, Vararg{Any}}, f, x...)
# map(f, x::AbstractArray{<:Number}) =
#     invoke(map, Tuple{Any, Union{AbstractArray, AbstractSet, AbstractDict}}, f, x)

# Implementation of sensitivities w.r.t. `broadcast`.
using Base.Broadcast
using Base.Broadcast: Broadcasted, broadcastable, broadcast_axes, broadcast_shape

struct NodeStyle{S} <: BroadcastStyle end

BroadcastStyle(::Type{<:Node{T}}) where {T} = NodeStyle{BroadcastStyle(T)}()

BroadcastStyle(::NodeStyle{S}, ::NodeStyle{S}) where {S} = NodeStyle{S}()
BroadcastStyle(::NodeStyle{S1}, ::NodeStyle{S2}) where {S1,S2} =
    NodeStyle{BroadcastStyle(S1, S2)}()
BroadcastStyle(::NodeStyle{S}, B::BroadcastStyle) where {S} =
    NodeStyle{BroadcastStyle(S, B)}()

Broadcast.broadcast_axes(x::Node) = broadcast_axes(x.val)
Broadcast.broadcastable(x::Node) = x

function Base.copy(bc::Broadcasted{<:NodeStyle})
    args = bc.args
    tape = getfield(args[findfirst(x -> x isa Node, args)], :tape)
    return Branch(broadcast, (bc.f, args...), tape)
end

"""
    broadcastsum!(f::Function, add::Bool, z, As...)

Broadcast f over As and reduce to z by summing. If add is true, then the result is added to
the current value of z, otherwise it is overwritten.
"""
function broadcastsum!(f::Function, add::Bool, z, As...)
    tmp_shape = broadcast_shape(map(size, As)...)
    if size(z) != tmp_shape
        tmp = Array{eltype(z)}(undef, tmp_shape)
        return sum!(z, broadcast!(f, tmp, As...), init=!add)
    else
        return add ?
            broadcast!((z, x...)->z + f(x...), z, z, As...) :
            broadcast!(f, z, As...)
    end
end

"""
    broadcastsum(f, add::Bool, z::AbstractArray, As...)

Allocating version of broadcastsum! specialised for Arrays.
"""
broadcastsum(f, add::Bool, z::AbstractArray, As...) =
    broadcastsum!(f, add, Array{eltype(z)}(undef, size(z)), As...)

"""
    broadcastsum(f, add::Bool, z::Number, As...)

Specialisation of broadcastsum to Number-sized outputs.
"""
function broadcastsum(f, add::Bool, z::Number, As...)
    tmp = Array{eltype(z)}(undef, broadcast_shape(map(size, As)...))
    return sum(broadcast!(f, tmp, As...)) + (add ? z : zero(z))
end

# Compute sensitivity w.r.t. the N^{th} input, N > 1.
∇(::typeof(broadcast), ::Type{Arg{N}}, p, y, ȳ, f, A::∇ArrayOrScalar...) where N =
    _∇(broadcast, Arg{N-1}, p, y, ȳ, f, A...)
_∇(::typeof(broadcast), ::Type{Arg{N}}, p, y, ȳ, f, A...) where N =
    hasmethod(∇, Tuple{typeof(f), Type{Arg{N}}, Any, Any, Any, map(eltype, A)...}) ?
        broadcastsum((yn, ȳn, xn...)->∇(f, Arg{N}, p, yn, ȳn, xn...), false, A[N], y, ȳ, A...) :
        broadcastsum((ȳn, xn...)->ȳn * fmad(f, xn, Val{N}), false, A[N], ȳ, A...)

# Addition.
import Base: +
@eval @explicit_intercepts $(Symbol("+")) Tuple{∇ArrayOrScalar, ∇ArrayOrScalar}
@inline ∇(::typeof(+), ::Type{Arg{1}}, p, z, z̄, x::∇ArrayOrScalar, y::∇ArrayOrScalar) =
    ∇(broadcast, Arg{2}, p, z, z̄, +, x, y)
@inline ∇(::typeof(+), ::Type{Arg{2}}, p, z, z̄, x::∇ArrayOrScalar, y::∇ArrayOrScalar) =
    ∇(broadcast, Arg{3}, p, z, z̄, +, x, y)

# Multiplication.
import Base: *
@eval @explicit_intercepts $(Symbol("*")) Tuple{∇ArrayOrScalar, ∇ArrayOrScalar}
@inline ∇(::typeof(*), ::Type{Arg{1}}, p, z, z̄, x::∇ArrayOrScalar, y::∇ArrayOrScalar) =
    ∇(broadcast, Arg{2}, p, z, z̄, *, x, y)
@inline ∇(::typeof(*), ::Type{Arg{2}}, p, z, z̄, x::∇ArrayOrScalar, y::∇ArrayOrScalar) =
    ∇(broadcast, Arg{3}, p, z, z̄, *, x, y)

# Subtraction.
import Base: -
@eval @explicit_intercepts $(Symbol("-")) Tuple{∇ArrayOrScalar, ∇ArrayOrScalar}
@inline ∇(::typeof(-), ::Type{Arg{1}}, p, z, z̄, x::∇ArrayOrScalar, y::∇ArrayOrScalar) =
    ∇(broadcast, Arg{2}, p, z, z̄, -, x, y)
@inline ∇(::typeof(-), ::Type{Arg{2}}, p, z, z̄, x::∇ArrayOrScalar, y::∇ArrayOrScalar) =
    ∇(broadcast, Arg{3}, p, z, z̄, -, x, y)

# Division from the right by a scalar.
import Base: /
@eval @explicit_intercepts $(Symbol("/")) Tuple{∇Array, ∇Scalar}
@inline ∇(::typeof(/), ::Type{Arg{1}}, p, z, z̄, x::∇ArrayOrScalar, y::∇ArrayOrScalar) =
    ∇(broadcast, Arg{2}, p, z, z̄, /, x, y)
@inline ∇(::typeof(/), ::Type{Arg{2}}, p, z, z̄, x::∇ArrayOrScalar, y::∇ArrayOrScalar) =
    ∇(broadcast, Arg{3}, p, z, z̄, /, x, y)

# Division from the left by a scalar.
import Base: \
@eval @explicit_intercepts $(Symbol("\\")) Tuple{∇Scalar, ∇Array}
@inline ∇(::typeof(\), ::Type{Arg{1}}, p, z, z̄, x::∇ArrayOrScalar, y::∇ArrayOrScalar) =
    ∇(broadcast, Arg{2}, p, z, z̄, \, x, y)
@inline ∇(::typeof(\), ::Type{Arg{2}}, p, z, z̄, x::∇ArrayOrScalar, y::∇ArrayOrScalar) =
    ∇(broadcast, Arg{3}, p, z, z̄, \, x, y)
