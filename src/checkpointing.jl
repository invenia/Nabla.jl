"""
    checkpoint(f, args::Tuple)

Treat `f` as a primitive, whose gradient is computed by running and forwards- and reverse-
pass of ∇ inside the outer pass. This can have the effect of significantly lowering the
peak memory usage of a gradient computation at the expense of additional computation.

HEALTH WARNING: If you close over differentiable objects in `f`, you will get incorrect
results. It is therefore very much recommended to verify your gradients using finite
differencing whenever code is checkpointed.
"""
checkpoint(f, args::Tuple) = f(args...)

@explicit_intercepts checkpoint Tuple{Any, Any} [false, true]

∇(::typeof(checkpoint), ::Type{Arg{1}}, p, y, ȳ, f, args::Tuple) = nothing
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
