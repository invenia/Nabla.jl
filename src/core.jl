using Cassette, BenchmarkTools
using Cassette: @context, tag, untag, enabletagging, overdub, OverdubInstead, istaggedtype,
    metadata, untagtype, Tagged
import Cassette: execute

import Base: push!

"""
    Op{Targs, Tkwargs, Tf, Tvalue}

The totality of a call to a (pure) primtive function `f` at `args` and `kwargs`,
producing `value`.
"""
struct Op{Targs, Tkwargs, Tf, Tvalue}
    args::Targs
    kwargs::Tkwargs
    f::Tf
    value::Tvalue
    function Op(f::Tf, args...; kwargs...) where Tf
        value = f(args...; kwargs...)
        return new{typeof(args), typeof(kwargs), Tf, typeof(value)}(args, kwargs, f, value)
    end
end

"""
    Tape

Used to keep track of operations that happen in a function call in a dynamic manner.
"""
const Tape = Vector{Tuple{Any, Tuple{Vararg{Int}}}}

# Fallback definition is false.
is_atom(args...) = false

# Execution context for ∇.jl (with a default dynamic tape).
@context ∇Ctx
const ∇Tagged{T} = Tagged{<:∇Ctx, T}
Cassette.metadatatype(::Type{<:∇Ctx}, ::Type{<:Any}) = Int
Cassette.metadatatype(::Type{<:∇Ctx}, ::DataType) = Int

@generated pos(x, ctx) = istaggedtype(x, ctx) ? :(metadata(x, ctx)) : :(-1)

const Arg{n} = Val{n}

"""
    execute(ctx::∇Ctx, f, args...; kwargs...)

If an operation and it's (non-keyword) argument constitute a primtive, then record it.
Otherwise just overdub.
"""
function execute(ctx::∇Ctx, f, args...; kwargs...)
    if is_atom(ctx, f, args...)
        args_, positions = map(x->untag(x, ctx), args), map(x->pos(x, ctx), args)
        op = Op(f, args_...; kwargs...)
        push!(ctx.metadata, (op, positions))
        return tag(op.value, ctx, length(ctx.metadata))
    else
        return OverdubInstead()
    end
end

"""
    forward(tape::Tape, f, args...)
    forward(f, args...)

Execute the function `f` at `args`, returning the result and the trace.
"""
function forward(tape::Tape, f, args...)

    # Push inputs onto the (forward) tape.
    foreach(arg->push!(tape, (arg, ())), args)

    # Create taggable context and tag stuff.
    ctx = enabletagging(∇Ctx(metadata=tape), f)
    tagged_args = map(arg->tag(arg[2], ctx, arg[1]), enumerate(args))

    # Execute the function, tracing all operations, returning the result and trace.
    y = overdub(ctx, f, tagged_args...)
    return untag(y, ctx)
end
@generated function is_atom(ctx::∇Ctx, ::typeof(forward), ::Tape, f, args...)
    return any(x->istaggedtype(x, ctx), args)
end

# Convenience functionality for the Tape-averse individual.
forward(f, args...) = forward(Tape(), f, args...)

"""
    ∇(::typeof(forward), ::Type{Arg{n}}, p::Tape, y, ȳ, tape::Tape, f, args...)

Reverse-mode sensitivity for `forward` - implemented like any other sensitivity. Only
applies for `n >= 2`.
"""
∇(::typeof(forward), ::Type{Arg{n}}, p::Vector, y, ȳ, tape::Tape, f, args...) where n = p[n-2]

# Perform the reverse pass.
preprocess(x...) = ()
function preprocess(::typeof(forward), y, ȳ, fwd_tape, f, args...)
    rvs_tape = Vector{Any}(undef, length(fwd_tape))
    rvs_tape[end] = ȳ
    for n in reverse(eachindex(rvs_tape))
        if isassigned(fwd_tape, n)
            ȳ, op, positions = rvs_tape[n], fwd_tape[n][1], fwd_tape[n][2]
            if op isa Op
                y, f, args, kwargs = op.value, op.f, op.args, op.kwargs
                pre = preprocess(f, y, ȳ, args...; kwargs...)
                for ((p, arg), pos) in zip(enumerate(args), positions)
                    if pos > 0
                        rvs_tape[pos] = isassigned(rvs_tape, pos) ?
                            ∇(rvs_tape[pos], f, Arg{p}, pre, y, ȳ, args...; kwargs...) :
                            ∇(f, Arg{p}, pre, y, ȳ, args...; kwargs...)
                    end
                end
            end
        end
    end
    return rvs_tape
end

"""
    ∇(f)

Returns a function `∇f = ∇(f)` which returns the gradient of `f` at the location evaluated.
"""
function ∇(f)
    return function(args...)

        # Perform the forwards-pass.
        tape = Tape()
        op = Op(forward, tape, f, args...)
        y = op.value

        # Perform the reverse-pass.
        ȳ = one(y)
        p = preprocess(forward, y, ȳ, tape, f, args...)
        return map(n->∇(forward, Arg{n + 2}, p, y, ȳ, tape, f, args...), eachindex(args))
    end
end

# As good a place as any to define this fallback.
∇(x̄, f, ::Type{Arg{N}}, args...) where N = x̄ + ∇(f, Arg{N}, args...)

# A collection of methods for initialising nested indexable containers to zero.
for (f_name, scalar_init, array_init) in
    zip((:zerod_container, :oned_container, :randned_container),
        (:zero, :one, nothing),
        (:zeros, :ones, nothing))
    if scalar_init !== nothing
        @eval @inline $f_name(x::Number) = $scalar_init(x)
    end
    if array_init !== nothing
        @eval @inline $f_name(x::AbstractArray{<:Real}) = $array_init(eltype(x), size(x))
    end
    eval(quote
        @inline $f_name(x::Tuple) = map($f_name, x)
        @inline function $f_name(x)
            y = Base.copy(x)
            for n in eachindex(y)
                @inbounds y[n] = $f_name(y[n])
            end
            return y
        end
    end)
end
@inline randned_container(x::Number) = randn(typeof(x))
@inline randned_container(x::AbstractArray{<:Real}) = randn(eltype(x), size(x)...)
for T in (:Diagonal, :UpperTriangular, :LowerTriangular)
    @eval @inline randned_container(x::$T{<:Real}) = $T(randn(eltype(x), size(x)...))
end

# CAN I CHANGE THIS TO USE ZYGOTE?

# Bare-bones FMAD implementation based on DualNumbers. Accepts a Tuple of args and returns
# a Tuple of gradients. Currently scales almost exactly linearly with the number of inputs.
# The coefficient of this scaling could be improved by implementing a version of DualNumbers
# which computes from multiple seeds at the same time.
function dual_call_expr(f, x::Type{<:Tuple}, ::Type{Type{Val{n}}}) where n
    dual_call = Expr(:call, :f)
    for m in 1:Base.length(x.parameters)
        push!(dual_call.args, n == m ? :(Dual(x[$m], 1)) : :(x[$m]))
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
