# Implementation of functionals (i.e. higher-order functions).
import Base.Broadcast.broadcast_shape
export mapreduce, mapreducedim, map, broadcast

# Intercepts for mapreduce.
accepted_add = :(Tuple{Function, typeof(+), AbstractArray{T} where T<:Real})
eval(DiffBase, add_intercept(:mapreduce, :(Base.mapreduce), accepted_add))
function ∇(
    ::typeof(mapreduce),
    ::Type{Arg{3}},
    p, y, ȳ, f,
    ::typeof(+),
    A::AbstractArray{T} where T<:Real,
)
    if needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real, Any})
        return Base.broadcast(An->ȳ * ∇(f, Arg{1}, An, f(An)), A)
    elseif !needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real})
        return Base.broadcast(An->ȳ * ∇(f, Arg{1}, An), A)
    else
        throw(error("Not implemented mapreduce sensitivities for general f. ($f)"))
    end
end

# Intercepts for mapreduce under multiplication.
accepted_mul = :(Tuple{Function, typeof(*), AbstractArray{T} where T<:Real})
eval(DiffBase, add_intercept(:mapreduce, :(Base.mapreduce), accepted_mul))
function ∇(
    ::typeof(mapreduce),
    ::Type{Arg{3}},
    p, y, ȳ, f,
    ::typeof(*),
    A::AbstractArray{T} where T<:Real,
)
    if needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real, Any})
        return Base.broadcast(An->ȳ * ∇(f, Arg{1}, An, f(An)), A)
    elseif !needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real})
        return Base.broadcast(An->ȳ * ∇(f, Arg{1}, An), A)
    else
        throw(error("Not implemented mapreduce sensitivities for general f. ($f)"))
    end
end

# Intercepts and sensitivities for mapreducedim.
accepted_wo_default = :(Tuple{Function, typeof(+), AbstractArray{T} where T<:Real, Any})
accepted_w_default = :(Tuple{Function, typeof(+), AbstractArray{T} where T<:Real, Any, Real})
eval(DiffBase, add_intercept(:mapreducedim, :(Base.mapreducedim), accepted_wo_default))
eval(DiffBase, add_intercept(:mapreducedim, :(Base.mapreducedim), accepted_w_default))

# Sensitivity w.r.t. mapreducedim.
function ∇(
    ::typeof(mapreducedim),
    ::Type{Arg{3}},
    p, y, ȳ, f,
    ::typeof(+),
    A::AbstractArray{T} where T<:Real,
    region,
)
    if needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real, Any})
        return Base.broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An, f(An)), A, ȳ)
    elseif !needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real})
        return Base.broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An), A, ȳ)
    else
        throw(error("Not implemented mapreducedim sensitivities for general f. ($f)"))
    end
end

# Sensitivity w.r.t. mapreducedim with default argument.
function ∇(
    ::typeof(mapreducedim),
    ::Type{Arg{3}},
    p, y, ȳ, f,
    ::typeof(+),
    A::AbstractArray{T} where T<:Real,
    region,
    v0::Real,
)
    if needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real, Any})
        return Base.broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An, f(An)), A, ȳ)
    elseif !needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real})
        return Base.broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An), A, ȳ)
    else
        throw(error("Not implemented mapreducedim sensitivities for general f. ($f)"))
    end
end

# Implementation of sensitivities w.r.t. `map`.
arr_type = AbstractArray{T} where T<:Real
accepted = :(Tuple{Function, Vararg{$(quot(arr_type))}})
eval(DiffBase, add_intercept(:map, :(Base.map), accepted))

# Make sure that we're not trying to differentiate the function being mapped.
∇(::typeof(map), ::Type{Arg{1}}, p, y, ȳ, f::Function, A::arr_type...) =
    throw(error("First argument of `map` is not differentiable."))

# Compute sensitivity w.r.t. the N^{th} input, N > 1.
∇(::typeof(map), ::Type{Arg{N}}, p, y, ȳ, f::Function, A::arr_type...) where N =
    method_exists(∇, Tuple{typeof(f), Type{Arg{N-1}}, Any, Any, Any, map(eltype, A)...}) ?
        Base.map((yn, ȳn, An...)->∇(f, Arg{N-1}, p, yn, ȳn, An...), y, ȳ, A...) :
        throw(error("Not implemented map sensitivities for general f. ($f)"))


# Implementation of sensitivities w.r.t. `broadcast`.
arg_type = Union{Real, AbstractArray{T} where T<:Real}
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
        tmp = Array(eltype(z), tmp_shape)
        return sum!(z, Base.broadcast!((x...)->f(x...), tmp, As...), init=!add)
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
        throw(error("Not implemented map sensitivities for general f. ($f)"))


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
const tp = Union{Real, RealArray, Node{T} where T<:Union{Real, RealArray}}
@generated Base.broadcast(f, A::Vararg{tp, N}) where N =
    any([issubtype(a, Node) for a in A]) ?
        :(DiffBase.broadcast(f, A...)) :
        :(invoke(Base.broadcast, Tuple{Any, Vararg{Any, N}}, f, A...))
@inline Base.broadcast(f, x::Union{Real, Node{T} where T<:Real}...) = f(x...)

# It is assumed that the cardinality of itr is relatively small in the methods below and]
# that there is therefore no need to optimise them.
# mapreduce(f, op, itr), mapreduce(f, op, v0, itr)

# # Reverse-mode sensitivities for each of the mapreducedim methods.
# eval(sensitivity(:(mapreducedim(f, op, A::AbstractArray, region)), ))
# eval(sensitivity(:(mapreducedim(f, op, A::AbstractArray, region, v0)), ))

# # Reverse-mode sensitivities for mapping operations involving tuples.
# eval(sensitivity(:(map(f, t::Tuple{Any})), ))
# eval(sensitivity(:(map(f, t::Tuple{Any, Any})), ))
# eval(sensitivity(:(map(f, t::Tuple{Any, Any, Any})), ))
# eval(sensitivity(:(map(f, t::Tuple)), ))
# eval(sensitivity(:(map(f, t::Any16)), ))

# eval(sensitivity(:(map(f, t::Tuple{Any}, s::Tuple{Any})), ))
# eval(sensitivity(:(map(f, t::Tuple{Any, Any}, s::Tuple{Any, Any})), ))
# eval(sensitivity(:(map(f, t::Tuple, s::Tuple)), ))
# eval(sensitivity(:(map(f, t::Any16, s::Any16)), ))
# eval(sensitivity(:(map(f, t1::Tuple, t2::Tuple, ts::Tuple...)), ))

# eval(sensitivity(:(map(f, t1::Any16, t2::Any16, ts::Any16...)), ))

# # Reverse-mode sensitivities for mapping operations involving numbers and arrays.
# eval(sensitivity(:(map(f, x::Number, ys::Number...)), ))
# eval(sensitivity(:(map(f, rowvecs::RowVector...)), ))
# eval(sensitivity(:(map(f, A::Union{AbstractArray, AbstractSet, Associative})), ))
# eval(sensitivity(:(map(f, A)), ))
# eval(sensitivity(:(map(f, iters...)), ))

# function map(f, A::Node{V}, B::Vararg{Node{T}} where V<:AbstractArray)
#     if method_exists(f, Tuple{eltype(A.val), map(x->eltype(x.val), B...)})
#         println("method exists!")
#     else
#         println("No such method exists... boooooo!")
#     end
# end

# # NOTE: FOR MAPPING OPERATIONS INVOLVING NON-ARRAY OBJECTS, NEED TO THINK ABOUT HOW THE
# # IMPLEMENTATION SHOULD WORK. THE OPTIMISATIONS WILL ONLY REDUCE MEMORY OVERHEAD SLIGHTLY IN
# # THE CASE OF SMALL TUPLES. IF ONE HAS A LARGE ARRAY OF ARRAYS, OPTIMISATIONS MIGHT MAKE
# # SENSE THOUGH. THIS REQUIRES A LOT OF CARE.

# # Sensitivities for broadcasted operations.
# eval(sensitivity(:(broadcast(::Base.*, x::Number, J::UniformScaling)), ))
# eval(sensitivity(:(broadcast(::Base.*, J::UniformScaling, x::Number)), ))

# eval(sensitivity(:(broadcast(::Base./, x::Number, J::UniformScaling)), ))
# eval(sensitivity(:(broadcast(::Base./, J::UniformScaling, x::Number)), ))

# eval(sensitivity(:(broadcast(f, x::Number...)), ))
# eval(sensitivity(:(broadcast(f, t::Tuple{Vararg{Any,N}}, ts::Tuple{Vararg{Any,N}}...)), ))
# eval(sensitivity(:(broadcast(f, rowvecs::Union{Number, RowVector}...)), ))
# eval(sensitivity(:(broadcast(f, A, Bs...)), ))
