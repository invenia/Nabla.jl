function add_kwargs!(ex::Expr; kwargs...)
    ex.head === :call || throw(ArgumentError("expression is not a function call"))
    isempty(ex.args) && throw(ArgumentError("expression body is empty"))
    if !isempty(kwargs)
        params = Expr(:parameters)
        for (name, value) in kwargs
            push!(params.args, Expr(:kw, name, value))
        end
        # Parameters need to come after the function name and before positional arguments
        if length(ex.args) == 1
            push!(ex.args, params)
        else
            insert!(ex.args, 2, params)
        end
    end
    ex
end

function get_union_call(foo::Symbol, type_tuple::Expr; kwargs...)
    # Get types from tuple and create a collection of symbols for use in the call.
    types = get_types(get_body(type_tuple))
    arg_names = [Symbol("x$j") for j in 1:length(types)]

    # Generate the call.
    typed_args = map((name, typ) -> :($name::$(unionise_type(typ))), arg_names, types)
    call = add_kwargs!(Expr(:call, foo, typed_args...); kwargs...)

    return replace_body(type_tuple, call), arg_names
end

"""
    unionise_type(tp::Union{Symbol, Expr})

Returns an expression for the type union of `tp` and `Node{<:tp}`. e.g.
`unionise_type(:Real)` returns `:(Union{Real, Node{<:Real}})`.
"""
function unionise_type(tp::Union{Symbol, Expr})
    (_tp, _info) = remove_vararg(tp)
    tp_clean = (isa(_tp, Expr) && _tp.head == Symbol("<:")) ? _tp.args[1] : _tp
    return replace_vararg(:(Union{$_tp, Node{<:$tp_clean}}), (_tp, _info))
end

"""
    node_type(tp::Union{Symbol, Expr})

Returns an expression for the `Node{<:tp}`. e.g.
`node_type(:Real)` returns `:(Node{<:Real}})`.

Correctly `Varargs{Real}` becomes `:(Varargs{Node{<:Real}})`

This is a lot like [`unionise_type`](ref) but it doesn't permit the original type anymore.
"""
function node_type(tp::Union{Symbol, Expr})
    (_tp, _info) = remove_vararg(tp)
    tp_clean = (isa(_tp, Expr) && _tp.head == Symbol("<:")) ? _tp.args[1] : _tp
    return replace_vararg(:(Node{<:$tp_clean}), (_tp, _info))
end


"""
    replace_body(unionall::Union{Symbol, Expr}, replacement::Union{Symbol, Expr})

Replace the body of an expression representing a `UnionAll`. e.g.
replace_body(:(Tuple{T, T} where T), :foo) returns the (nonsensical) expression
:(foo where T). If `unionall` is a `Symbol`, then `replacement` is returned.
"""
replace_body(unionall::Expr, replacement::Union{Symbol, Expr}) =
    unionall.head == :where ?
        Expr(:where, replace_body(unionall.args[1], replacement), unionall.args[2:end]...) :
        replacement
replace_body(::Symbol, replacement::Union{Symbol, Expr}) = replacement

"""
    get_body(x::Union{Symbol, Expr})

Get the body from an expression representing a `UnionAll`. e.g. :(Tuple{T, T} where T)
returns :(Tuple{T, T}). If `x` is a `Symbol`, then `x` is returned unaltered.
"""
get_body(unionall::Expr) = unionall.head == :where ? get_body(unionall.args[1]) : unionall
get_body(body::Symbol) = body

"""
    get_types(type_tuple::Expr)

Return the types from a type-tuple expression. e.g. :(Tuple{Float64, Real}) returns a
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

        # This is a bit ugly:
        # handle interally `where N` from `typ = :(Vararg{FOO, N} where N)` which results in
        # `body = :(Vararg{FOO, N})` and `new_type = Foo where N`, we don't need to keep it
        # at all, the `where N` wasn't doing anything to begin with, so we just strip it out
        if Meta.isexpr(new_typ, :where, 2) && Meta.isexpr(body, :curly, 3)
            @assert body.args[1] == :Vararg
            T = body.args[2]
            N = body.args[3]
            if new_typ.args == [T, N]
                body = :(Vararg{T})
                new_typ = T
            end
        end

        vararg_info = length(body.args) == 3 ? body.args[3] : :Vararg
        return new_typ, vararg_info
    else
        return (typ, :nothing)
    end
end

"""
    replace_vararg(typ::SymOrExpr, vararg_info::Tuple)

Convert `typ` to the `Vararg` containing elements of type `typ` specified by
`vararg_info`, which should be a `Tuple` returned from `remove_vararg`.
"""
replace_vararg(typ::SymOrExpr, vararg_info::Tuple) =
    vararg_info[2] == :nothing ?
        typ :
        vararg_info[2] == :no_N || vararg_info[2] == :Vararg ?  #TODO: :no_N is impossible now?
            replace_body(typ, :(Vararg{$(get_body(typ))})) :
            replace_body(typ, :(Vararg{$(get_body(typ)), $(vararg_info[2])}))

"""
    parse_kwargs(nt_expr) -> NamedTuple

Accepts an expression containing a `NamedTuple` literal and parses it into a `NamedTuple`
with expressions as values.
"""
function parse_kwargs(nt_expr)
    if isempty(nt_expr.args) || nt_expr == Expr(:call, :NamedTuple)
        return NamedTuple()
    end

    first_arg = first(nt_expr.args)
    if first_arg isa Expr && first_arg.head == :parameters
        return parse_kwargs_parameters(nt_expr)
    elseif first_arg isa Expr && first_arg.head == :(=)
        return parse_kwargs_tuple(nt_expr)
    else
        throw(ArgumentError("Unsupported expression $nt_expr for kwargs;"
            * " they must be passed as a NamedTuple literal"))
    end
end

function parse_kwargs_tuple(tup_expr)
    nt_names = Tuple(first(ex.args) for ex in tup_expr.args)
    return NamedTuple{nt_names}(last(ex.args) for ex in tup_expr.args)
end

function parse_kwargs_parameters(param_tuple_expr)
    # code is the same even though the inner args are also different expression types
    # (:kw vs :())
    return parse_kwargs_tuple(param_tuple_expr.args[1])
end

"""
    parse_is_node(bool_array_expr) -> Vector{Bool}

Accepts an expression containing a `Vector{Bool}` literal and parses it into a
`Vector{Bool}`.
"""
function parse_is_node(bool_array_expr)
    if bool_array_expr.head != :vect
        throw(ArgumentError("Unsupported expression $bool_array_expr for is_node; "
            * "it must be passed as a `Vector{Bool}` literal (e.g., `[true, false]`)"))
    end

    return collect(Bool, bool_array_expr.args)
end
