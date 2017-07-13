import Base.Meta.quot
export @differentiable, add_intercept, Arg, add_∇, add_∇!, ∇, preprocess, intercepts,
    @generated, keys, values, intercept_names

const intercept_names = Set()

"""
    add_intercept(foo::Symbol, base_foo::Expr)

Add an intercept for the function from module `mod` whose name is `foo` such that calls to\\
it from within an `@differentiable` block of code or module will be tracked.
"""
function add_intercept(foo::Symbol, base_foo::Expr, type_tuple::SymOrExpr)

    # If not previously defined a method for foo, define a catch-all.
    add_name = Expr(:call, :push!, :intercept_names, quot(foo))
    catch_all = !in(foo, intercept_names) ? catch_all_expr(foo, base_foo) : :nothing
    original = !in(foo, intercept_names) ? original_expr(foo, base_foo) : nothing

    if type_tuple != :nothing
        call, arg_names = get_union_call(foo, type_tuple)
        body = get_body(foo, base_foo, type_tuple, arg_names)
        specific_method = Expr(:macrocall, Symbol("@generated"), Expr(:function, call, body))
    else
        specific_method = :nothing
    end
    return Expr(:block, add_name, catch_all, original, specific_method)
end
add_intercept(foo::Symbol, base_foo::Expr, accepted_types::Type{T} where T<:Tuple) =
    add_intercept(foo, base_foo, Vector{Type{T} where T<:Tuple}([accepted_types]))

"""
    get_body(foo::Symbol, base_foo::Symbol, type_tuple::Expr, arg_names::Vector)

Get the body of the @generated function which is used to intercept hte invokations\\
specified by type_tuple.
"""
function get_body(foo::Symbol, base_foo::SymOrExpr, type_tuple::Expr, arg_names::Vector)
    arg_tuple = any(isa_vararg.(get_types(get_body(type_tuple)))) ?
        Expr(:tuple, arg_names[1:end-1]..., Expr(Symbol("..."), arg_names[end])) :
        Expr(:tuple, arg_names...)
    quot_arg_names = [quot(arg_name) for arg_name in arg_names]

    dots = Symbol("...")
    return Expr(:block,
        Expr(Symbol("="), :x, Expr(:tuple, Expr(dots, Expr(:vect, arg_names...)))),
        Expr(Symbol("="), :x_syms, Expr(:tuple, Expr(dots, Expr(:vect, quot_arg_names...)))),
        Expr(Symbol("="), :is_node, :([any(issubtype.(xj, Node)) for xj in x])),
        Expr(:return,
            Expr(:if, Expr(:call, :any, :is_node),
                :(DiffCore.branch_expr(
                    $(quot(foo)),
                    is_node,
                    x,
                    x_syms,
                    $(quot(arg_tuple)),
                )),
                :(DiffCore.bypass_diff_expr($(quot(base_foo)), x, x_syms)),
            )
        )
    )
end

"""
    branch_expr(foo::Symbol, is_node::Vector{Bool}, x_symbol, arg_tuple::Expr)

Generate an expression to call Branch.
"""
function branch_expr(
    foo::Symbol,
    is_node::Vector{Bool},
    x::Tuple,
    syms::NTuple{N, Symbol} where N,
    arg_tuple::Expr,
)
    # tape = Expr(:call, :getfield, syms[findfirst(is_node)], quot(:tape))
    return Expr(:call, :Branch, foo, arg_tuple, tape_expr(x, syms, is_node))
end

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
    bypass_diff_expr(base_foo::Symbol, x::Tuple, syms::Vector)

Generate an expression which calls call_with_originals with appropriately vararg-ed args.
"""
function bypass_diff_expr(base_foo::SymOrExpr, x::Tuple, syms::NTuple{N, Symbol} where N)
    args, vararg = isa(x[end], Tuple) ?
        (syms[1:end-1], (Expr(Symbol("..."), syms[end]),)) :
        (syms, ())
    return Expr(:call, :(DiffCore.call_with_originals), base_foo, args..., vararg...)
end

catch_all_expr(foo::Symbol, base_foo::Expr) = :($foo(x...) = $base_foo(x...))
original_expr(foo::Symbol, base_foo::Expr) = :(get_original(x::typeof($foo)) = $base_foo)
@inline call_with_originals(f::Function, args...) = f(Base.map(get_original, args)...)

function get_union_call(foo::Symbol, type_tuple::Expr, arg_types::Vector{Symbol})

    # Get type info from tuple and declare a collection of symbols for use in the call.
    types = get_types(get_body(type_tuple))
    arg_names = [gensym() for _ in types]
    arg_names = [Symbol("x$j") for j in 1:length(types)]

    # Remove strip out Vararg stuff, compute unioned types, and re-add Vararg stuff.
    type_info = remove_vararg.(types)
    unioned_types = [:(Union{$typ, Node{$par} where $par <: $typ})
        for (typ, par) in zip(getindex.(type_info, 1), arg_types)]
    vararged_types = replace_vararg.(unioned_types, type_info)

    # Generate the call.
    typed_args = [:($name::$typ) for (name, typ) in zip(arg_names, vararged_types)]
    return replace_body(type_tuple, Expr(:call, foo, typed_args...)), arg_names
end
get_union_call(foo::Symbol, type_tuple::Expr) =
    get_union_call(foo, type_tuple, [gensym() for _ in get_types(get_body(type_tuple))])

replace_body(unionall::Expr, replacement::Union{Symbol, Expr}) =
    unionall.head == :where ?
        Expr(:where, replace_body(unionall.args[1], replacement), unionall.args[2:end]...) :
        replacement
replace_body(::Symbol, replacement::Union{Symbol, Expr}) = replacement

get_body(unionall::Expr) = unionall.head == :where ? get_body(unionall.args[1]) : unionall
get_body(body::Symbol) = body

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
    remove_vararg(typ::Expr)
    remove_vararg(typ::Symbol)

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

"""
    add_∇(f::Function, arg::Type{Arg{N}} where N, δ::Function)

Convenience functions for declaring extra ∇ reverse-mode sensitivity implementations.\\
`f` is the function to which the sensitivity should be added, and `arg` specifies the\\
argument w.r.t. which the sensitivity should be added. `δ` is the implementation of the \\
sensitivity. Doesn't return anything, and calls eval from within 
e.g.\\
    add_∇(+, Arg{1}, (p, x, y, z, z̄)->z̄)\\
will make the sensitivity w.r.t. the first argument of `+` the lambda function provided.\\
"""
function add_∇(f::Function, arg::Type{Arg{N}}, δ::Function) where N
    global ∇(::typeof(f), ::Type{Arg{N}}, p, y, ȳ, x...) = δ(p, y, ȳ, x...)
end

function add_∇!(f::Function, arg::Type{Arg{N}}, δ::Function) where N
    global ∇(x̄, ::typeof(f), ::Type{Arg{N}}, p, y, ȳ, x...) = δ(x̄, p, y, ȳ, x...)
end

"""
    preprocess(::Function, args...)

Default implementation of preprocess returns an empty Tuple. Individual sensitivity\\
implementations should add methods specific to their use case. The output is passed\\
in to `∇` as the 3rd or 4th argument in the new-x̄ and update-x̄ cases respectively.
"""
@inline preprocess(::Function, args...) = ()

"""
    needs_output(::Function)

Returns a bool determining whether the particular function in question requires access to\\
its output to compute it's gradient. Defaults to true. Useful for making efficient\\
implementations of `mapreduce` and `mapreducedim`.
"""
@inline needs_output(::Function) = true

"""
    get_original(x)

Returns the "original" version of `x`. This is useful to handle the situation in which a\\
function `g` is passed as an argument to another function `f` and it is necessary for the\\
module in which `f` is originally defined to dispatch on the type of `g`. In this case\\
if `g` was redefined for the purposes of auto-diff, the type will no longer match that\\
which was specified in the module in which it was originally defined and dispatch will fail.
"""
@inline get_original(x) = x
