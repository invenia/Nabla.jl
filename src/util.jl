export generate_primitive

"""
    generate_primitive(
        f::Symbol,
        typepars::Vector,
        x::Vector{Symbol},
        x̄::Vector{Symbol},
        xtypes::Vector,
        diffs::Vector{Bool},
        y::Symbol,
        ȳ::Symbol,
        x̄0::Vector,
        x̄_update::Vector,
        preprocess::SymOrExpr=:nothing
    )

Inputs:\\\
`f` - function to generate method for.\\\
`typepars` - the parametric type information for the method. eg. [:T, :(V <: AbstractArray)]\\\
`x` - inputs to `f` in the forward-pass.\\\
`x̄` - current values of reverse-mode sensitivities of the corresponding elements of `x`.\\\
`xtypes` - the type of each element of `x` respectively.\\\
`diffs` - indicates which argument are differentiable.
`y` - output of `f` from the forward-pass.\\\
`ȳ` - reverse-mode sensitivities of the corresponding elements of `y`.\\\
`x̄0` - expressions to create sensitivites `x̄` if currently uninitialised.\\\
`x̄_update` - expressions to update the sensitivites `x̄`.\\\
"""
function generate_primitive(
    f::Symbol,
    typepars::Vector,
    x::Vector{Symbol},
    x̄::Vector{Symbol},
    xtypes::Vector,
    diffs::Vector{Bool},
    y::Symbol,
    ȳ::Symbol,
    x̄0::Vector,
    x̄_update::Vector,
    preprocess::SymOrExpr=:nothing
)
    # Ensure that lengths are consistent and that typepars, xtypes and sensitivities only
    # contain Symbols and Exprs.
    length(x) != length(x̄) && throw(ArgumentError("length(x) != length(x̄)."))
    length(x) != length(xtypes) && throw(ArgumentError("length(x) != length(xtypes)."))
    length(x) != length(x̄0) && throw(ArgumentError("length(x) != length(x̄0)."))
    length(x) != length(x̄_update) && throw(ArgumentError("length(x) != length(x̄_update)."))
    isallSymOrExpr(typepars) || throw(ArgumentError("typepars has non SymOrExpr element."))
    isallSymOrExpr(xtypes) || throw(ArgumentError("xtypes has non SymOrExpr element."))
    isallSymOrExpr(x̄0) || throw(ArgumentError("x̄0 has non SymOrExpr element."))
    isallSymOrExpr(x̄_update) || throw(ArgumentError("x̄_update has non SymOrExpr element."))

    # Symbols for the tape and indices into `tape` to get `x̄`.
    tape = gensym()
    x̄id = [gensym() for n in eachindex(x̄)]

    # Generate the method signature. This is a :call Expr.
    typedname = Expr(:curly, f, typepars...)
    tape_arg = Expr(:(::), tape, :Tape)
    x_typed = typedargs(x, xtypes)
    xid̄_typed = typedargs(x̄id, :Int)
    signature = Expr(:call, typedname, tape_arg, y, ȳ, x_typed..., xid̄_typed...)

    # Create function body and add in pre-processing steps if they are provided.
    body = Expr(:block)
    preprocess != :nothing && push!(body.args, preprocess)

    # For each argument in `x`, add code to compute the reverse-mode sensitivity, updating
    # the existing value if present, otherwise creating a new value. Always assign in the
    # end.
    for n in eachindex(x)
        if diffs[n]
            tape_index = :($tape.tape[$(x̄id[n])])
            update_x̄ = Expr(:block, Expr(:(=), x̄[n], tape_index), x̄_update[n])
            push!(body.args,
                Expr(:if, :($(x̄id[n]) > 0), Expr(:block,
                    Expr(:if, :(isdefined($tape.tape, $(x̄id[n]))), update_x̄, x̄0[n]),
                    Expr(:(=), tape_index, x̄[n]))))
        end
    end

    push!(body.args, Expr(:return, :nothing))
    eval(Expr(:function, signature, body))
    primitive(f, typepars, xtypes, diffs)
end


"""
    typedargs(args::Vector{Symbol}, types::Vector)

Compute a vector of Exprs of the form `arg_name::arg_type`, where `args` contains each\\\
`arg_name` and `types` contains each `arg_type`.\\\
"""
function typedargs(args::Vector{Symbol}, types::Vector)
    isallSymOrExpr(types) || throw(ArgumentError("types has non SymOrExpr element."))
    return [Expr(:(::), a, t) for (a, t) in zip(args, types)]
end


"""
    typedargs(args::Vector{Symbol}, t::SymOrExpr)

Compute a vector of Exprs of the form `arg_name::t`, where `args` contains each `arg_name`.\\\
"""
typedargs(args::Vector{Symbol}, t::SymOrExpr) = [Expr(:(::), a, t) for a in args]


"""
    isallSymOrExpr(v::Vector)

`true` if each element of `v` is a `Symbol` or `Expr`, otherwise `false`.\\\
"""
isallSymOrExpr(v::Vector) = all([typeof(n) <: SymOrExpr for n in v])
