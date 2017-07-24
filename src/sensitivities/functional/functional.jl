# Implementation of functionals (i.e. higher-order functions).
import Base.Broadcast.broadcast_shape, Nabla.DiffBase.fmad
export map, broadcast

# Implementation of sensitivities w.r.t. `map`.
arr_type = AbstractArray{<:Real}
accepted = :(Tuple{Function, Vararg{$(quot(arr_type))}})
eval(DiffBase, add_intercept(:map, :(Base.map), accepted))

# Make sure that we're not trying to differentiate the function being mapped.
∇(::typeof(map), ::Type{Arg{1}}, p, y, ȳ, f::Function, A::arr_type...) =
    throw(error("First argument of `map` is not differentiable."))

# Compute sensitivity w.r.t. the N^{th} input, N > 1.
∇(::typeof(map), ::Type{Arg{N}}, p, y, ȳ, f::Function, A::arr_type...) where N =
    method_exists(∇, Tuple{typeof(f), Type{Arg{N-1}}, Any, Any, Any, map(eltype, A)...}) ?
        Base.map((yn, ȳn, An...)->∇(f, Arg{N-1}, p, yn, ȳn, An...), y, ȳ, A...) :
        Base.map((ȳn, An...)->ȳn * fmad(f, An, Val{N-1}), ȳ, A...)


# Implementation of sensitivities w.r.t. `broadcast`.
arg_type = Union{Real, AbstractArray{<:Real}}
accepted = :(Tuple{Function, Vararg{$(quot(arg_type))}})
eval(DiffBase, add_intercept(:broadcast, :(Base.broadcast), accepted))

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
∇(::typeof(broadcast), ::Type{Arg{N}}, p, y, ȳ, f::Function, A::arg_type...) where N =
    method_exists(∇, Tuple{typeof(f), Type{Arg{N-1}}, Any, Any, Any, map(eltype, A)...}) ?
        broadcastsum((yn, ȳn, xn...)->∇(f, Arg{N-1}, p, yn, ȳn, xn...), false, A[N-1], y, ȳ, A...) :
        broadcastsum((ȳn, xn...)->ȳn * fmad(f, xn, Val{N-1}), false, A[N-1], ȳ, A...)

# Scalar-array operations without dots. All just implemented in terms of broadcast.
const ArrayOrReal = Union{Real, AbstractArray{T} where T<:Real}
accepted = :(Tuple{ArrayOrReal, ArrayOrReal})

# Addition.
eval(DiffBase, add_intercept(Symbol("+"), :(getfield(Base, Symbol("+"))), accepted))
@inline ∇(::typeof(+), ::Type{Arg{1}}, p, z, z̄, x::ArrayOrReal, y::ArrayOrReal) =
    ∇(broadcast, Arg{2}, p, z, z̄, +, x, y)
@inline ∇(::typeof(+), ::Type{Arg{2}}, p, z, z̄, x::ArrayOrReal, y::ArrayOrReal) =
    ∇(broadcast, Arg{3}, p, z, z̄, +, x, y)

# Multiplication.
eval(DiffBase, add_intercept(Symbol("*"), :(getfield(Base, Symbol("*"))), accepted))
@inline ∇(::typeof(*), ::Type{Arg{1}}, p, z, z̄, x::ArrayOrReal, y::ArrayOrReal) =
    ∇(broadcast, Arg{2}, p, z, z̄, *, x, y)
@inline ∇(::typeof(*), ::Type{Arg{2}}, p, z, z̄, x::ArrayOrReal, y::ArrayOrReal) =
    ∇(broadcast, Arg{3}, p, z, z̄, *, x, y)

# Subtraction.
eval(DiffBase, add_intercept(Symbol("-"), :(getfield(Base, Symbol("-"))), accepted))
@inline ∇(::typeof(-), ::Type{Arg{1}}, p, z, z̄, x::ArrayOrReal, y::ArrayOrReal) =
    ∇(broadcast, Arg{2}, p, z, z̄, -, x, y)
@inline ∇(::typeof(-), ::Type{Arg{2}}, p, z, z̄, x::ArrayOrReal, y::ArrayOrReal) =
    ∇(broadcast, Arg{3}, p, z, z̄, -, x, y)

# Division from the right by a scalar.
accepted = :(Tuple{AbstractArray{T} where T<:Real, Real})
eval(DiffBase, add_intercept(Symbol("/"), :(getfield(Base, Symbol("/"))), accepted))
@inline ∇(::typeof(/), ::Type{Arg{1}}, p, z, z̄, x::ArrayOrReal, y::ArrayOrReal) =
    ∇(broadcast, Arg{2}, p, z, z̄, /, x, y)
@inline ∇(::typeof(/), ::Type{Arg{2}}, p, z, z̄, x::ArrayOrReal, y::ArrayOrReal) =
    ∇(broadcast, Arg{3}, p, z, z̄, /, x, y)

# Division from the left by a scalar.
accepted = :(Tuple{Real, AbstractArray{T} where T<:Real})
eval(DiffBase, add_intercept(Symbol("\\"), :(getfield(Base, Symbol("\\"))), accepted))
@inline ∇(::typeof(\), ::Type{Arg{1}}, p, z, z̄, x::ArrayOrReal, y::ArrayOrReal) =
    ∇(broadcast, Arg{2}, p, z, z̄, \, x, y)
@inline ∇(::typeof(\), ::Type{Arg{2}}, p, z, z̄, x::ArrayOrReal, y::ArrayOrReal) =
    ∇(broadcast, Arg{3}, p, z, z̄, \, x, y)


# We have to add some methods to Base to ensure that dispatch happens correctly when using
# the dot notation.
const tp = Union{Real, RealArray, Node{<:Union{Real, RealArray}}}
@generated Base.broadcast(f, A::Vararg{tp, N}) where N =
    any([issubtype(a, Node) for a in A]) ?
        :(DiffBase.broadcast(f, A...)) :
        :(invoke(Base.broadcast, Tuple{Any, Vararg{Any, N}}, f, A...))
@inline Base.broadcast(f, x::Union{Real, Node{<:Real}}...) = f(x...)

# Bare-bones FMAD implementation based on DualNumbers. Accepts a Tuple of args and returns
# a Tuple of gradients. Currently scales almost exactly linearly with the number of inputs.
# The coefficient of this scaling could be improved by implementing a version of DualNumbers
# which computes from multiple seeds at the same time.
function dual_call_expr(f, x::Type{<:Tuple}, ::Type{Type{Val{n}}}) where n
    dual_call = Expr(:call, :f)
    for m in 1:Base.length(x.parameters)
        push!(dual_call.args, :(Dual(x[$m], $(Base.isequal(n, m) ? 1 : 0))))
    end
    return :(dualpart($dual_call))
end
@generated fmad(f, x, n) = dual_call_expr(f, x, n)
function fmad_expr(f, x::Type{<:Tuple})
    body = Expr(:tuple)
    for n in 1:Base.length(x.parameters)
        push!(body.args, dual_call_expr(f, x, Type{Val{n}}))
    end
    return body
end
@generated fmad(f, x) = fmad_expr(f, x)
