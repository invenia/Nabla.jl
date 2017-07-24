function get_union_call(foo::Symbol, type_tuple::Expr)

    # Get types from tuple and create a collection of symbols for use in the call.
    types = get_types(get_body(type_tuple))
    arg_names = [Symbol("x$j") for j in 1:length(types)]

    # Generate the call.
    typed_args = [:($name::$typ) for (name, typ) in zip(arg_names, unionise_type.(types))]
    return replace_body(type_tuple, Expr(:call, foo, typed_args...)), arg_names
end

"""
    unionise_type(tp::Union{Symbol, Expr})

Returns an expression for the type union of `tp` and `Node{<:tp}`. e.g.\\
`unionise_type(:Real)` returns `:(Union{Real, Node{<:Real}})`.
"""
function unionise_type(tp::Union{Symbol, Expr})
    (_tp, _info) = remove_vararg(tp)
    tp_clean = (isa(_tp, Expr) && _tp.head == Symbol("<:")) ? _tp.args[1] : _tp
    return replace_vararg(:(Union{$_tp, Node{<:$tp_clean}}), (_tp, _info))
end

"""
    replace_body(unionall::Union{Symbol, Expr}, replacement::Union{Symbol, Expr})

Replace the body of an expression representing a `UnionAll`. e.g.\\
replace_body(:(Tuple{T, T} where T), :foo) returns the (nonsensical) expression\\
:(foo where T). If `unionall` is a `Symbol`, then `replacement` is returned.
"""
replace_body(unionall::Expr, replacement::Union{Symbol, Expr}) =
    unionall.head == :where ?
        Expr(:where, replace_body(unionall.args[1], replacement), unionall.args[2:end]...) :
        replacement
replace_body(::Symbol, replacement::Union{Symbol, Expr}) = replacement

"""
    get_body(x::Union{Symbol, Expr})

Get the body from an expression representing a `UnionAll`. e.g. :(Tuple{T, T} where T)\\
returns :(Tuple{T, T}). If `x` is a `Symbol`, then `x` is returned unaltered.
"""
get_body(unionall::Expr) = unionall.head == :where ? get_body(unionall.args[1]) : unionall
get_body(body::Symbol) = body

"""
    get_types(type_tuple::Expr)

Return the types from a type-tuple expression. e.g. :(Tuple{Float64, Real}) returns a\\
vector [:Float64, :Real]
"""
get_types(type_tuple::Expr) = type_tuple.args[2:end]

"""
    isa_vararg(symbol_or_expr)

Returns a bool indicating whether `symbol_or_expr` is a `Vararg`.
"""
isa_vararg(sym::Symbol) = (sym == :Vararg)
isa_vararg(expr::Expr) =
    (expr.head == :curly && expr.args[1] == :Vararg) ?
        true :
        expr.head == :where ? isa_vararg(expr.args[1]) : false

"""
    remove_vararg(typ::Union{Symbol, Expr})

Return the type contained by the `typ` if it's a `Vararg` and the `N` parameter if provided.
"""
remove_vararg(typ::Symbol) = isa_vararg(typ) ? (:Any, :Vararg) : (typ, :nothing)
function remove_vararg(typ::Expr)
    if isa_vararg(typ)
        body = get_body(typ)
        new_typ = replace_body(typ, body.args[2])
        vararg_info = length(body.args) == 3 ? body.args[3] : :Vararg
        return new_typ, vararg_info
    else
        return (typ, :nothing)
    end
end

"""
    replace_vararg(typ::SymOrExpr, vararg_info::Tuple)

Convert `typ` to the `Vararg` containing elements of type `typ` specified by\\
`vararg_info`, which should be a `Tuple` returned from `remove_vararg`.
"""
replace_vararg(typ::SymOrExpr, vararg_info::Tuple) =
    vararg_info[2] == :nothing ?
        typ :
        vararg_info[2] == :no_N || vararg_info[2] == :Vararg ?
            replace_body(typ, :(Vararg{$(get_body(typ))})) :
            replace_body(typ, :(Vararg{$(get_body(typ)), $(vararg_info[2])}))
