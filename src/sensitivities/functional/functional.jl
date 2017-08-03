# Implementation of functionals (i.e. higher-order functions).
import Base.Broadcast.broadcast_shape

# Implementation of sensitivities w.r.t. `map`.
import Base.map
@explicit_intercepts map Tuple{Any, ∇RealArray} [false, true]
@union_intercepts map Tuple{Any, Vararg{∇RealArray}} Tuple{Any, Vararg}

# Make sure that we're not trying to differentiate the function being mapped.
∇(::typeof(map), ::Type{Arg{1}}, p, y, ȳ, f::Function, A::∇RealArray...) =
    throw(error("First argument of `map` is not differentiable."))

# Compute sensitivity w.r.t. the N^{th} input, N > 1.
∇(::typeof(map), ::Type{Arg{N}}, p, y, ȳ, f::Function, A::∇RealArray...) where N =
    method_exists(∇, Tuple{typeof(f), Type{Arg{N-1}}, Any, Any, Any, map(eltype, A)...}) ?
        Base.map((yn, ȳn, An...)->∇(f, Arg{N-1}, p, yn, ȳn, An...), y, ȳ, A...) :
        Base.map((ȳn, An...)->ȳn * fmad(f, An, Val{N-1}), ȳ, A...)

# Implementation of sensitivities w.r.t. `broadcast`.
import Base.broadcast
@union_intercepts broadcast Tuple{Any, Vararg{∇Real}} Tuple{Any, Vararg{Number}}
@explicit_intercepts broadcast Tuple{Any, Any} [false, true]
@union_intercepts broadcast Tuple{Any, Vararg{ArrayOr∇Real}} Tuple{Any, Any, Vararg}

# Make sure that we're not trying to differentiate the function being mapped.
∇(::typeof(broadcast), ::Type{Arg{1}}, p, y, ȳ, f, A, B...) =
    throw(error("First argument of broadcast is not differentiable."))

"""
    broadcastsum!(f::Function, add::Bool, z, As...)

Broadcast f over As and reduce to z by summing. If add is true, then the result is added to
the current value of z, otherwise it is overwritten.
"""
function broadcastsum!(f::Function, add::Bool, z, As...)
    tmp_shape = broadcast_shape(map(size, As)...)
    if size(z) != tmp_shape
        tmp = Array{eltype(z)}(tmp_shape)
        return Base.sum!(z, Base.broadcast!((x...)->f(x...), tmp, As...), init=!add)
    else
        return add ?
            Base.broadcast!((z, x...)->z + f(x...), z, z, As...) :
            Base.broadcast!((x...)->f(x...), z, As...)
    end
end

"""
    broadcastsum(f::Function, add::Bool, z::AbstractArray, As...)

Allocating version of broadcastsum! specialised for Arrays.
"""
function broadcastsum(f::Function, add::Bool, z::AbstractArray, As...)
    return broadcastsum!(f, add, similar(z), As...)
end

"""
    broadcastsum(f::Function, add::Bool, z::Number, As...)

Specialisation of broadcastsum to Number-sized outputs.
"""
function broadcastsum(f::Function, add::Bool, z::Number, As...)
    tmp = Array{eltype(z)}(broadcast_shape(map(size, As)...))
    return Base.sum(Base.broadcast!((x...)->f(x...), tmp, As...)) + (add ? z : zero(z))
end

# Compute sensitivity w.r.t. the N^{th} input, N > 1.
∇(::typeof(broadcast), ::Type{Arg{N}}, p, y, ȳ, f::Function, A::ArrayOr∇Real...) where N =
    method_exists(∇, Tuple{typeof(f), Type{Arg{N-1}}, Any, Any, Any, map(eltype, A)...}) ?
        broadcastsum((yn, ȳn, xn...)->∇(f, Arg{N-1}, p, yn, ȳn, xn...), false, A[N-1], y, ȳ, A...) :
        broadcastsum((ȳn, xn...)->ȳn * fmad(f, xn, Val{N-1}), false, A[N-1], ȳ, A...)

# Addition.
@eval @explicit_intercepts $(Symbol("+")) Tuple{ArrayOr∇Real, ArrayOr∇Real}
@inline ∇(::typeof(+), ::Type{Arg{1}}, p, z, z̄, x::ArrayOr∇Real, y::ArrayOr∇Real) =
    ∇(broadcast, Arg{2}, p, z, z̄, +, x, y)
@inline ∇(::typeof(+), ::Type{Arg{2}}, p, z, z̄, x::ArrayOr∇Real, y::ArrayOr∇Real) =
    ∇(broadcast, Arg{3}, p, z, z̄, +, x, y)

# Multiplication.
@eval @explicit_intercepts $(Symbol("*")) Tuple{ArrayOr∇Real, ArrayOr∇Real}
@inline ∇(::typeof(*), ::Type{Arg{1}}, p, z, z̄, x::ArrayOr∇Real, y::ArrayOr∇Real) =
    ∇(broadcast, Arg{2}, p, z, z̄, *, x, y)
@inline ∇(::typeof(*), ::Type{Arg{2}}, p, z, z̄, x::ArrayOr∇Real, y::ArrayOr∇Real) =
    ∇(broadcast, Arg{3}, p, z, z̄, *, x, y)

# Subtraction.
@eval @explicit_intercepts $(Symbol("-")) Tuple{ArrayOr∇Real, ArrayOr∇Real}
@inline ∇(::typeof(-), ::Type{Arg{1}}, p, z, z̄, x::ArrayOr∇Real, y::ArrayOr∇Real) =
    ∇(broadcast, Arg{2}, p, z, z̄, -, x, y)
@inline ∇(::typeof(-), ::Type{Arg{2}}, p, z, z̄, x::ArrayOr∇Real, y::ArrayOr∇Real) =
    ∇(broadcast, Arg{3}, p, z, z̄, -, x, y)

# Division from the right by a scalar.
@eval @explicit_intercepts $(Symbol("/")) Tuple{∇RealArray, ∇Real}
@inline ∇(::typeof(/), ::Type{Arg{1}}, p, z, z̄, x::ArrayOr∇Real, y::ArrayOr∇Real) =
    ∇(broadcast, Arg{2}, p, z, z̄, /, x, y)
@inline ∇(::typeof(/), ::Type{Arg{2}}, p, z, z̄, x::ArrayOr∇Real, y::ArrayOr∇Real) =
    ∇(broadcast, Arg{3}, p, z, z̄, /, x, y)

# Division from the left by a scalar.
@eval @explicit_intercepts $(Symbol("\\")) Tuple{∇Real, ∇RealArray}
@inline ∇(::typeof(\), ::Type{Arg{1}}, p, z, z̄, x::ArrayOr∇Real, y::ArrayOr∇Real) =
    ∇(broadcast, Arg{2}, p, z, z̄, \, x, y)
@inline ∇(::typeof(\), ::Type{Arg{2}}, p, z, z̄, x::ArrayOr∇Real, y::ArrayOr∇Real) =
    ∇(broadcast, Arg{3}, p, z, z̄, \, x, y)
