"""
    checkpoint

Do checkpointing.
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
