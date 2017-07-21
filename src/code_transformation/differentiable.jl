import Base.include_string
export @differentiable

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
    foreach(arg->push!(body.args, esc(arg)), code.args)
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
    unionise_sig(code)

`code` should be a `tuple`, `call` or `where` expression (containing a `tuple` or `call`).\\
Returns a signature which accepts `Node`s.
"""
unionise_sig(code::Symbol) = code
function unionise_sig(code::Expr)
    body = get_body(code)
    if body.head == :tuple
        return Expr(:tuple, unionise_arg.(body.args)...)
    elseif body.head == :call
        return Expr(:call, body.args[1], unionise_arg.(body.args[2:end])...)
    elseif body.head == Symbol("::")
        return unionise_arg(body)
    else
        throw(error("Not sure what to do."))
    end
end

"""
    make_accept_nodes(code)

Return transformed code in which all function definitions are guaranteed to accept nodes as
arguments. This should not affect the existing functionality of the code.
"""
function make_accept_nodes end

# If we get a symbol then we cannot have found a function definition, so ignore it.
make_accept_nodes(code) = code

# Recurse through an expression, bottoming out if we find a function definition.
function make_accept_nodes(code::Expr)
    if code.head in (:function, Symbol("->"))
        return Expr(code.head, unionise_sig(code.args[1]), code.args[2])
    elseif code.head == Symbol("=") &&
        (get_body(code.args[1]).head == :tuple || get_body(code.args[1]).head isa Symbol)
        return Expr(code.head, unionise_sig(code.args[1]), code.args[2])
    else
        return Expr(code.head, [make_accept_nodes(arg) for arg in code.args]...)
    end
end
