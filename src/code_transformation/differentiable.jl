import Base.include_string
export @differentiable, @unionise

"""
    @differentiable name code

Make a block of code differentiable. See documentation for details.
"""
macro differentiable(name::Symbol, code::Expr)
    return differentiable(esc(name), code)
end

function differentiable(name, code)
    body = Expr(:block)
    push!(body.args, :(import Base))
    push!(body.args, :(import Nabla))
    push!(body.args, :(using Nabla.DiffCore))
    push!(body.args, :(using Nabla.DiffBase))
    push!(body.args, :(include_string(base_include_str(DiffCore.intercept_names))))
    foreach(arg->push!(body.args, esc(:(@unionise $arg))), code.args)
    return Expr(:toplevel, Expr(:module, false, name, body))
end

"""
    base_include_str(exclude_names)

Create a string which, when included (via include_string), imports all of the names from\\
Base which are not present in `exclude_names`.
"""
function base_include_str(exclude_names)
    base_include = "import Base: "
    foreach(x->!in(x, exclude_names) && (base_include *= String(x) * ", "), names(Base))
    return base_include[1:end-2]
end

"""
    unionise_arg(arg)

Return an expression in which the argument expression `arg` is replaced with an argument\\
whos type admits `Node`s.
"""
unionise_arg(arg::Symbol) = arg
function unionise_arg(arg::Expr)
    arg.head != Symbol("::") && throw(error("Unrecognised argument."))
    return Expr(Symbol("::"), arg.args[1:end-1]..., unionise_type(arg.args[end]))
end

"""
    get_quote_body(code)

Return the body of a quoted expression or symbol. There is slightly different behaviour\\
depending upon whether a symbol or expression is quoted and how exactly it is quoted,\\
thus we dispatch on whether code is a `:quote` expression or a `QuoteNode` object.
"""
get_quote_body(code::Expr) = code.args[1]
get_quote_body(code::QuoteNode) = code.value

"""
    unionise_eval(code::Expr)

Unionise the code inside a call to `eval`, such that when the `eval` call actually occurs\\
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

Unionise the code in a call to @eval, such that when the `eval` call actually occurs, the\\
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

`code` should be a `tuple`, `call` or `where` expression (containing a `tuple` or `call`).\\
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
    else
        throw(error("Not sure what to do."))
    end
    return replace_body(code, new_body)
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
    else
        return Expr(code.head, [unionise(arg) for arg in code.args]...)
    end
end

"""
    @unionise code

Transform code such that each function definition accepts `Node` objects as arguments,\\
without effecting dispatch in other ways.
"""
macro unionise(code)
    return esc(unionise(code))
end

"""
    unionised_include(path::AbstractString)

Apply the @unionise macro to all of the code loaded in an `include` call.
"""
unionised_include(path::AbstractString) =
    include_string("@unionise begin " * readstring(open(path)) * " end")

"""
    unionise_include(code::Expr)

Return an expression which calls `unionised_include` instead of simply `include`.
"""
unionise_include(code::Expr) = Expr(:call, :unionised_include, code.args[2])
