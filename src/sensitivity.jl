import Base.Meta.quot
export Arg, add_∇, add_∇!, ∇, preprocess, @explicit_intercepts, @union_intercepts

"""
    @union_intercepts f type_tuple invoke_type_tuple

Interception strategy based on adding a method to `f` which accepts the union of each of
the types specified by `type_tuple`. If none of the arguments are `Node`s then the method
of `f` specified by `invoke_type_tuple` is invoked.
"""
macro union_intercepts(f::Symbol, type_tuple::Expr, invoke_type_tuple::Expr)
    return esc(union_intercepts(f, type_tuple, invoke_type_tuple))
end

"""
    union_intercepts(f::Symbol, type_tuple::Expr, invoke_type_tuple::Expr)

The work-horse for `@union_intercepts`.
"""
function union_intercepts(f::Symbol, type_tuple::Expr, invoke_type_tuple::Expr)
    call, arg_names = get_union_call(f, type_tuple)
    body = get_body(f, type_tuple, arg_names, invoke_type_tuple)
    return Expr(:macrocall, Symbol("@generated"), Expr(:function, call, body))
end

"""
    @explicit_intercepts(f::Symbol, type_tuple::Expr, is_node::Expr)
    @explicit_intercepts(f::Symbol, type_tuple::Expr)

Create a collection of methods which intecept the function calls to `f` in which at least
one argument is a `Node`. Types of arguments are specified by the type tuple expression
in `type_tuple`. If there are arguments which are not differentiable, they can be
specified by providing a boolean vector `is_node` which indicates those arguments that are
differentiable with `true` values and those which are not as `false`.
"""
macro explicit_intercepts(f::SymOrExpr, type_tuple::Expr, is_node::Expr)
    return esc(explicit_intercepts(f, type_tuple, eval(is_node)))
end
macro explicit_intercepts(f::SymOrExpr, type_tuple::Expr)
    is_node = [true for _ in get_types(get_body(type_tuple))]
    return esc(explicit_intercepts(f, type_tuple, is_node))
end

"""
    explicit_intercepts(f::Symbol, types::Expr, is_node::Vector)

Return a `:block` expression which evaluates to declare all of the combinations of methods
that could be required to catch if a `Node` is ever passed to the function specified in
`expr`.
"""
function explicit_intercepts(f::SymOrExpr, types::Expr, is_node::Vector{Bool})
    function explicit_intercepts_(states::Vector{Bool})
        if length(states) == length(is_node)
            return any(states .== true) ? boxed_method(f, types, states) : []
        else
            return vcat(
                explicit_intercepts_(vcat(states, false)),
                is_node[length(states) + 1] ? explicit_intercepts_(vcat(states, true)) : []
            )
        end
    end
    return Expr(:block, explicit_intercepts_(Vector{Bool}())...)
end

"""
    boxed_method(
        f::SymOrExpr,
        type_tuple::Expr,
        is_node::Vector{Bool},
        arg_names::Vector{Symbol},
    )

Construct a method of the Function `f`, whose argument's types are specified by
`type_tuple`. Arguments which are potentially `Node`s should be indicated by `true` values
in `is_node`.
"""
function boxed_method(
    f::SymOrExpr,
    type_tuple::Expr,
    is_node::Vector{Bool},
    arg_names::Vector{Symbol},
)
    # Get the argument types and create the function call.
    types = get_types(get_body(type_tuple))
    noded_types = [node ? :(Node{<:$tp}) : tp for (node, tp) in zip(is_node, types)]
    call = replace_body(type_tuple, get_sig(f, arg_names, noded_types))

    # Construct body of call.
    tuple_expr = Expr(:tuple, arg_names...)
    tape_expr = Expr(:call, :getfield, arg_names[findfirst(is_node)], quot(:tape))
    body = Expr(:call, :Branch, f, tuple_expr, tape_expr)

    # Combine call signature with the body to create a new function.
    return Expr(Symbol("="), call, body)
end
boxed_method(f, t, n) = boxed_method(f, t, n, [gensym() for _ in n])

"""
    get_sig(f::SymOrExpr, arg_names::Vector{Symbol}, types::Vector)

Generate a function signature for `f` in which the arguments, whose names are `arg_names`,
specified by the `true` entires of `is_node` have type `Node`. The other arguments have
types specified by `types`.
"""
get_sig(f::SymOrExpr, arg_names::Vector{Symbol}, types::Vector) =
    Expr(:call, f, [Expr(Symbol("::"), nm, tp) for (nm, tp) in zip(arg_names, types)]...)

"""
    get_body(foo::Symbol, type_tuple::Expr, arg_names::Vector, invoke_type_tuple::Expr)

Get the body of the @generated function which is used to intercept the invokations
specified by type_tuple.
"""
function get_body(
    foo::Symbol,
    type_tuple::Expr,
    arg_names::Vector,
    invoke_type_tuple::Expr
)
    arg_tuple = any(isa_vararg.(get_types(get_body(type_tuple)))) ?
        Expr(:tuple, arg_names[1:end-1]..., Expr(Symbol("..."), arg_names[end])) :
        Expr(:tuple, arg_names...)
    sym_arg_tuple = any(isa_vararg.(get_types(get_body(type_tuple)))) ?
        Expr(:tuple, quot.(arg_names[1:end-1])...,
                           quot(Expr(Symbol("..."), arg_names[end]))) :
        Expr(:tuple, quot.(arg_names)...)
    quot_arg_names = [quot(arg_name) for arg_name in arg_names]

    dots = Symbol("...")
    args_dotted = Expr(dots, Expr(:vect, arg_names...))
    args_dotted_quot = Expr(dots, Expr(:vect, quot_arg_names...))
    return Expr(:block,
        Expr(Symbol("="), :x, Expr(:tuple, args_dotted)),
        Expr(Symbol("="), :x_syms, Expr(:tuple, args_dotted_quot)),
        Expr(Symbol("="), :x_dots, sym_arg_tuple),
        Expr(Symbol("="), :is_node, :([any(issubtype.(xj, Node)) for xj in x])),
        Expr(:return,
            Expr(:if, Expr(:call, :any, :is_node),
                :(Nabla.branch_expr(
                    $(quot(foo)),
                    is_node,
                    x,
                    x_syms,
                    $(quot(arg_tuple)),
                )),
                :(Nabla.invoke_expr($(quot(foo)), $(quot(invoke_type_tuple)), x_dots))
            )
        )
    )
end

"""
    branch_expr(foo::Symbol, is_node::Vector{Bool}, x::Tuple, arg_tuple::Expr)

Generate an expression to call Branch.
"""
function branch_expr(
    foo::Symbol,
    is_node::Vector{Bool},
    x::Tuple,
    syms::NTuple{<:Any, Symbol},
    arg_tuple::Expr,
)
    return Expr(:call, :Branch, foo, arg_tuple, tape_expr(x, syms, is_node))
end

invoke_expr(f::Symbol, invoke_tuple::Expr, arg_syms) =
    Expr(:call, :invoke, f, invoke_tuple, arg_syms...)

"""
    tape_expr(x::Tuple, syms::NTuple{N, Symbol} where N, is_node::Vector{Bool})

Get an expression which will obtain the tape from a Node object in `x`.
"""
function tape_expr(x::Tuple, syms::NTuple{N, Symbol} where N, is_node::Vector{Bool})
    idx = findfirst(is_node)
    if idx == length(is_node) && isa(x[end], Tuple)
        node_idx = findfirst([issubtype(varg, Node) for varg in x[end]])
        return Expr(:call, :getfield, Expr(:ref, syms[end], node_idx), quot(:tape))
    else
        return Expr(:call, :getfield, syms[idx], quot(:tape))
    end
end

"""
    preprocess(::Function, args...)

Default implementation of preprocess returns an empty Tuple. Individual sensitivity
implementations should add methods specific to their use case. The output is passed
in to `∇` as the 3rd or 4th argument in the new-x̄ and update-x̄ cases respectively.
"""
@inline preprocess(::Any, args...) = ()
