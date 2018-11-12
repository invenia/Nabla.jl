using Cassette, BenchmarkTools, InteractiveUtils
using Cassette: @context, tag, untag, enabletagging, overdub, OverdubInstead, istaggedtype,
    metadata, untagtype, Tagged
import Cassette: execute

import Base: push!, show

export ∇

"""
    Op{Tf, Tvalue, Targs, Tkwargs}

The totality of a call to a (pure) primtive function `f` at `args` and `kwargs`,
producing `value`.
"""
struct Op{Tf, Tvalue, Targs, Tkwargs}
    f::Tf
    value::Tvalue
    args::Targs
    kwargs::Tkwargs
    function Op(f::Tf, args...; kwargs...) where Tf
        value = f(args...; kwargs...)
        return new{Tf, typeof(value), typeof(args), typeof(kwargs)}(f, value, args, kwargs)
    end
    function Op(value::T) where T
        return new{Nothing, T, Nothing, Nothing}(nothing, value, nothing, nothing)
    end
end

# Alias for distinguishing between leaves and branches.
const Leaf = Op{Nothing}
is_leaf(::Leaf) = true
is_leaf(::Op) = false

value(op::Op) = op.value

function show(io::IO, op::Op)
    print(io, "$(op.f) (Op)")
end
function show(io::IO, mime::MIME"text/plain", op::Op)
    println("Op where")
    println("f = $(op.f)")
    println("y = $(value(op))")
    println("args = $(op.args)")
    print("kwargs = $(op.kwargs)")
end

function show(io::IO, op::Leaf)
    print(io, "$(value(op)) (Leaf)")
end

struct TapePair
    op::Op
    positions::Tuple{Vararg{Int}}
end
operation(pair::TapePair) = pair.op
positions(pair::TapePair) = pair.positions

const Tape = Vector{TapePair}

function show(io::IO, mime::MIME"text/plain", tape::Tape)
    if length(tape) == 0
        print("0-element Tape")
    else
        println("$(length(tape))-element Tape:")
        for (n, pair) in enumerate(tape)
            str = " %$n = $pair"
            (n == length(tape) ? print : println)(str)
        end
    end
end

# Fallback definition is false
is_atom(args...) = false

# Execution context for ∇.jl (with a default dynamic tape).
@context ∇Ctx
const ∇MaybeTagged{T} = Union{T, Tagged{C, T} where C}
Cassette.metadatatype(::Type{<:∇Ctx}, ::Type{<:Any}) = Int
Cassette.metadatatype(::Type{<:∇Ctx}, ::DataType) = Int

@generated position(x, ctx) = istaggedtype(x, ctx) ? :(metadata(x, ctx)) : :(-1)

const Arg{n} = Val{n}

"""
    execute(ctx::∇Ctx, f, args...; kwargs...)

If an operation and it's (non-keyword) argument constitute a primtive, then record it.
Otherwise just overdub.
"""
function execute(ctx::∇Ctx, f, args...; kwargs...)
    if is_atom(ctx, f, args...)
        args_, positions = map(x->untag(x, ctx), args), map(x->position(x, ctx), args)
        op = Op(f, args_...; kwargs...)
        push!(ctx.metadata, TapePair(op, positions))
        return tag(value(op), ctx, length(ctx.metadata))
    else
        return OverdubInstead()
    end
end

"""
    forward!(tape::Tape, f, args...)

Execute the function `f` at `args`, returning the result and the trace.
"""
function forward!(tape::Tape, f, args...)

    # Push inputs onto the (forward) tape.
    l0 = length(tape)
    foreach(arg->push!(tape, TapePair(Op(arg), ())), args)

    # Create taggable context and tag stuff.
    ctx = enabletagging(∇Ctx(metadata=tape), f)
    tagged_args = (map(arg->tag(arg[2], ctx, l0 + arg[1]), enumerate(args))...,)

    # Execute the function, tracing all operations, returning the result.
    tmp = execute(ctx, f, tagged_args...)
    return untag(tmp isa OverdubInstead ? overdub(ctx, f, tagged_args...) : tmp, ctx)
end
@generated function is_atom(ctx::∇Ctx, ::typeof(forward!), ::Tape, f, args...)
    return any(x->istaggedtype(x, ctx), args)
end

"""
    forward(f, args...)

Execute the function `f` at `args`, tracing each of the operations performed.
"""
forward(f, args...) = forward!(Tape(), f, args...)

"""
    ∇(::typeof(forward!), ::Type{Arg{n}}, p::Tape, y, ȳ, tape::Tape, f, args...)

Reverse-mode sensitivity for `forward!` - implemented like any other sensitivity. Only
applies for `n >= 2`.
"""
∇(::typeof(forward!), ::Type{Arg{n}}, p::Vector, y, ȳ, tape::Tape, f, args...) where n = p[n-2]

preprocess(x...) = ()

# Perform the reverse pass.
function get_rvs_tape(fwd_tape, ȳ)
    rvs_tape = Vector{Any}(undef, length(fwd_tape))
    rvs_tape[end] = ȳ
    return rvs_tape
end
function preprocess(::typeof(forward!), y, ȳ, fwd_tape::Tape, f, args...)
    return preprocess!(get_rvs_tape(fwd_tape, ȳ), forward!, y, ȳ, fwd_tape, f, args...)
end
function preprocess!(rvs_tape::Vector{Any}, ::typeof(forward!), y, ȳ, fwd_tape, f, args...)
    for n in reverse(eachindex(rvs_tape))
        if isassigned(fwd_tape, n)
            ȳ, op, pos = rvs_tape[n], operation(fwd_tape[n]), positions(fwd_tape[n])
            if !isa(op, Leaf)
                y, f, args, kwargs = value(op), op.f, op.args, op.kwargs
                pre = preprocess(f, y, ȳ, args...; kwargs...)
                for ((p, arg), pos) in zip(enumerate(args), pos)
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
        op = Op(forward!, tape, f, args...)
        y = value(op)

        # Perform the reverse-pass.
        ȳ = one(y)
        p = preprocess(forward!, y, ȳ, tape, f, args...)
        return (map(n->∇(forward!, Arg{n + 2}, p, y, ȳ, tape, f, args...), eachindex(args))...,)
    end
end

# Ugly function intended for internal use only.
function __forward(f, args...)

    # Perform forward pass with tracking.
    tape = Tape()
    op = Op(forward!, tape, f, args...)
    y = value(op)

    # Return result of forwards pass and closure to compute adjoint.
    return y, function(ȳ)
        p = preprocess(forward!, y, ȳ, tape, f, args...)
        return (map(n->∇(forward!, Arg{n + 2}, p, y, ȳ, tape, f, args...), eachindex(args))...,)
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
