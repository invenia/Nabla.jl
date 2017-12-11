export @unionise

"""
    unionise_arg(arg)

Return an expression in which the argument expression `arg` is replaced with an argument
whos type admits `Node`s.
"""
unionise_arg(arg::Symbol) = arg
unionise_arg(arg::Expr) =
    arg.head == Symbol("::") ?
        Expr(Symbol("::"), arg.args[1:end-1]..., unionise_type(arg.args[end])) :
        arg.head == Symbol("...") ?
            Expr(Symbol("..."), unionise_arg(arg.args[1])) :
            throw(error("Unrecognised argument in arg ($arg)."))

"""
    unionise_subtype(arg::Union{Symbol, Expr})

Equivalent to `unionise_arg`, but replacing `::` with `<:`.
"""
unionise_subtype(arg::Symbol) = arg
unionise_subtype(arg::Expr) =
    arg.head == Symbol("<:") ?
        Expr(Symbol("<:"), arg.args[1:end-1]..., unionise_type(arg.args[end])) :
        throw(error("Unrecognised argument in arg ($arg)."))

"""
    get_quote_body(code)

Return the body of a quoted expression or symbol. There is slightly different behaviour
depending upon whether a symbol or expression is quoted and how exactly it is quoted,
thus we dispatch on whether code is a `:quote` expression or a `QuoteNode` object.
"""
get_quote_body(code::Expr) = code.args[1]
get_quote_body(code::QuoteNode) = code.value

"""
    unionise_eval(code::Expr)

Unionise the code inside a call to `eval`, such that when the `eval` call actually occurs
the code inside will be unionised.
"""
function unionise_eval(code::Expr)
    body = Expr(:macrocall, Symbol("@unionise"), deepcopy(get_quote_body(code.args[end])))
    return length(code.args) == 3 ?
        Expr(:call, :eval, deepcopy(code.args[2]), quot(body)) :
        Expr(:call, :eval, quot(body))
end

"""
    unionise_macro_eval(code::Expr)

Unionise the code in a call to @eval, such that when the `eval` call actually occurs, the
code inside will be unionised.
"""
function unionise_macro_eval(code::Expr)
    body = Expr(:macrocall, Symbol("@unionise"), deepcopy(code.args[end]))
    return length(code.args) == 3 ?
        Expr(:macrocall, Symbol("@eval"), deepcopy(code.args[2]), body) :
        Expr(:macrocall, Symbol("@eval"), body)
end

"""
    unionise_sig(code)

`code` should be a `tuple`, `call` or `where` expression (containing a `tuple` or `call`).
Returns a signature which accepts `Node`s.
"""
unionise_sig(code::Symbol) = code
function unionise_sig(code::Expr)
    body = get_body(code)
    if body.head == :tuple
        new_body = Expr(:tuple, unionise_arg.(body.args)...)
    elseif body.head == :call
        new_body = Expr(:call, body.args[1], unionise_arg.(body.args[2:end])...)
    elseif body.head == Symbol("::")
        new_body = unionise_arg(body)
    end
    return replace_body(code, new_body)
end

"""
    unionise_struct(code)

`code` should be an `Expr` containing the definition of a type. The type will only be
changed if it is parametric. That is, if you wish to be able to differentiate through a
user-defined type, it must contain only `Any`s and parametric types.
"""
function unionise_struct(code::Expr)
    name = code.args[2]
    if name isa Expr && name.head == :curly
        curly = Expr(:curly, name.args[1], unionise_subtype.(name.args[2:end])...)
        return Expr(:type, code.args[1], curly, code.args[3])
    else
        return code
    end
end

"""
    unionise(code)

Return transformed code in which all function definitions are guaranteed to accept nodes as
arguments. This should not affect the existing functionality of the code.
"""
function unionise end

# If we get a symbol then we cannot have found a function definition, so ignore it.
unionise(code) = code

# Recurse through an expression, bottoming out if we find a function definition or a
# quoted expression to be `eval`-ed.
function unionise(code::Expr)
    if code.head in (:function, Symbol("->"))
        return Expr(code.head, unionise_sig(code.args[1]), code.args[2])
    elseif code.head == Symbol("=") && !isa(code.args[1], Symbol) &&
        (get_body(code.args[1]).head == :tuple || get_body(code.args[1]).head isa Symbol)
        return Expr(code.head, unionise_sig(code.args[1]), code.args[2])
    elseif code.head == :call && code.args[1] == :eval
        return unionise_eval(code)
    elseif code.head == :macrocall && code.args[1] == Symbol("@eval")
        return unionise_macro_eval(code)
    elseif code.head == :type
        return unionise_struct(code)
    else
        return Expr(code.head, [unionise(arg) for arg in code.args]...)
    end
end

"""
    @unionise code

Transform code such that each function definition accepts `Node` objects as arguments,
without effecting dispatch in other ways.
"""
macro unionise(code)
    return esc(unionise(code))
end
