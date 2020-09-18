# Implementation of functionals (i.e. higher-order functions).

# Implementation of sensitivities w.r.t. `map`.

# Build `@explicit_intercepts`-like calls for `map` with a variable number of arguments.
# We set an arbitrary cutoff of 10 input arrays, which should be sufficient, as mapping
# over more than 3 arrays simultaneously is (anecdotally) pretty uncommon. Imposing this
# cutoff alleviates the need to overwrite a Base method to call `invoke` due to method
# ambiguities, though even that can be insufficient to resolve them, as in issue #136.
const ArrayOrNode = Union{AbstractArray, Node{<:AbstractArray}}
for nargs = 1:10
    name = Symbol(:A, nargs)
    args = Expr[:($(Symbol(:A, i))::$ArrayOrNode) for i = 1:nargs-1]
    push!(args, :($name::Node{<:AbstractArray}))
    @eval function Base.map(f, $(args...), As::$ArrayOrNode...)
        Branch(map, (f, $(map(i->Symbol(:A, i), 1:nargs)...), As...), getfield($name, :tape))
    end
end

# Compute sensitivity w.r.t. the N^{th} input, N > 1.
∇(::typeof(map), ::Type{Arg{N}}, p, y, ȳ, f::Function, A::∇Array...) where N =
    _∇(map, Arg{N-1}, p, y, ȳ, f, A...)
_∇(::typeof(map), arg::Type{Arg{N}}, p, y, ȳ, f::Function, A::∇Array...) where N =
        map((ȳn, An...)->ȳn * fmad(f, An, Val{N}), ȳ, A...)

# Implementation of sensitivities w.r.t. `broadcast`.
using Base.Broadcast
using Base.Broadcast: Broadcasted, broadcastable, broadcast_axes, combine_axes, result_style

struct NodeStyle{S} <: BroadcastStyle end

Base.BroadcastStyle(::Type{<:Node{T}}) where {T} = NodeStyle{BroadcastStyle(T)}()

Base.BroadcastStyle(::NodeStyle{S}, ::NodeStyle{S}) where {S} = NodeStyle{S}()
function Base.BroadcastStyle(::NodeStyle{S1}, ::NodeStyle{S2}) where {S1,S2}
    promoted = result_style(S1, S2)
    promoted isa Broadcast.Unknown ? promoted : NodeStyle{promoted}()
end
function Base.BroadcastStyle(::NodeStyle{S}, B::BroadcastStyle) where {S}
    promoted = result_style(S, B)
    promoted isa Broadcast.Unknown ? promoted : NodeStyle{promoted}()
end

Broadcast.broadcast_axes(x::Node) = broadcast_axes(unbox(x))
Broadcast.broadcastable(x::Node) = x

# eagerly construct a Branch when encountering a Node in broadcasting
function Broadcast.broadcasted(::NodeStyle, f, args...)
    tape = getfield(args[findfirst(x -> x isa Node, args)], :tape)
    return Branch(broadcast, (f, args...), tape)
end

"""
    broadcastsum!(f::Function, add::Bool, z, As...)

Broadcast f over As and reduce to z by summing. If add is true, then the result is added to
the current value of z, otherwise it is overwritten.
"""
function broadcastsum!(f::Function, add::Bool, z, As...)
    tmp_shape = map(length, combine_axes(As...))
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
    tmp = Array{eltype(z)}(undef, map(length, combine_axes(As...)))
    return sum(broadcast!(f, tmp, As...)) + (add ? z : zero(z))
end

broadcastsum(f, add::Bool, z::Ref{<:Number}, As...) = broadcastsum(f, add, z[], As...)

# Compute sensitivity w.r.t. the N^{th} input, N > 1.
∇(::typeof(broadcast), ::Type{Arg{N}}, p, y, ȳ, f, A...) where N =
    _∇(broadcast, Arg{N-1}, p, y, ȳ, f, A...)
_∇(::typeof(broadcast), ::Type{Arg{N}}, p, y, ȳ, f, A...) where N =
        broadcastsum((ȳn, xn...)->ȳn * fmad(f, xn, Val{N}), false, A[N], ȳ, A...)

# Division from the right by a scalar.
import Base: /
@eval @explicit_intercepts $(Symbol("/")) Tuple{∇Array, ∇Scalar}
@inline ∇(::typeof(/), ::Type{Arg{1}}, p, z, z̄, x::∇Scalar, y::∇Array) =
    ∇(broadcast, Arg{2}, p, z, z̄, /, x, y)
@inline ∇(::typeof(/), ::Type{Arg{2}}, p, z, z̄, x::∇Scalar, y::∇Array) =
    ∇(broadcast, Arg{3}, p, z, z̄, /, x, y)

# Division from the left by a scalar.
import Base: \
@eval @explicit_intercepts $(Symbol("\\")) Tuple{∇Scalar, ∇Array}
@inline ∇(::typeof(\), ::Type{Arg{1}}, p, z, z̄, x::∇Array, y::∇Scalar) =
    ∇(broadcast, Arg{2}, p, z, z̄, \, x, y)
@inline ∇(::typeof(\), ::Type{Arg{2}}, p, z, z̄, x::∇Array, y::∇Scalar) =
    ∇(broadcast, Arg{3}, p, z, z̄, \, x, y)
