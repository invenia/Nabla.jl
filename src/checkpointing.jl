"""
    checkpoint(f, args::Tuple)

Treat `f` as a primitive, whose gradient is computed by running and forwards- and reverse-
pass of ∇ inside the outer pass. This can have the effect of significantly lowering the
peak memory usage of a gradient computation at the expense of additional computation.

`f` musn't have any fields. In particular, this precludes `f` from being a closure.
"""
@generated function checkpoint(f, args::Tuple)
    fieldcount(f) > 0 && error("f mustn't have fields.")
    return :(f(args...))
end
@generated function checkpoint(f, args::Tuple, kwargs::NamedTuple)
    fieldcount(f) > 0 && error("f mustn't have fields. Can not checkpoint a closure or functor. Please reformulate as a plain function that does not close over any parameters.")
    return :(f(args...; kwargs...))
end

@explicit_intercepts checkpoint Tuple{Any, Tuple} [false, true]
@explicit_intercepts checkpoint Tuple{Any, Tuple, NamedTuple} [false, true, false]

function ∇(::typeof(checkpoint), ::Type{Arg{2}}, p, y, ȳ, f, args::Tuple)
    args_ = Leaf.(Tape(), args)
    y = f(args_...)
    if y isa Node
        ∇f = ∇(y, ȳ)
        ∇args = map(args_, args) do arg_, arg
            isassigned(∇f, arg_) ? ∇f[arg_] : zero(arg)
        end
    else
        ∇args = zero.(args)
    end
    return ∇args
end

function ∇(
    ::typeof(checkpoint),
    ::Type{Arg{2}},
    p, y, ȳ, f,
    args::Tuple,
    kwargs::NamedTuple,
)
    args_ = Leaf.(Tape(), args)
    y = f(args_...; kwargs...)
    if y isa Node
        ∇f = ∇(y, ȳ)
        ∇args = map(args_, args) do arg_, arg
            isassigned(∇f, arg_) ? ∇f[arg_] : zero(arg)
        end
    else
        ∇args = zero.(args)
    end
    return ∇args
end
