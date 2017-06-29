export @get_gen_intercepts, branch_expr, preprocess, ∇, Arg, change_unionall_body

macro get_gen_intercepts()
    out = quote
        function gen_intercepts(expr::Expr, diffs::NTuple{N, Bool} where N=())
            (expr, call, name, args) = $(pkg_name).parse_expr(expr)
            if in(name, $(pkg_name).use_fallback)
                throw(error("$name is in the fallback dictionary, so is currently unsupported."))
            end
            tuple_type = eval($(pkg_name).to_tuple_type(expr))
            ms = methods(eval(name))
            any_subtypes = any([tuple_type <: m.sig for m in ms])
            any_matches = any([tuple_type == m.sig for m in ms])
            edge = any_subtypes && !any_matches ?
                AutoGrad2.edge_intercept(expr) : :nothing
            node = AutoGrad2.union_intercept(expr)
            return Expr(:block, edge, node)
        end
    end
    return esc(out)
end
@get_gen_intercepts

"""
    union_intercept(expr::Expr)

Generates the code for a method which is designed to intercept the usual control flow of a\\
program when Node objects are encountered where differentiable arguments may occur. `expr`\\
specifies the signature of the method to be generated and should be a `:call` or `:where`\\
expression and correspond to a valid method signature. e.g.\\
    foo(x::Real, y::Real)\\
or\\
    foo{T<:Real}(x::T, y::T)\\
or\\
    foo(x::T, y::T) where T<:Real\\
Any argument of `foo` which is to be differentiable (i.e. could be boxed in a Node object)\\
must have a which is not `Any`. For example\\
    foo(x::Real, y)\\
would produce unreliable behaviour if it happens to be the case that y<:Node. Conversely,\\
if an argument is not meant to be differentiable, then it is fine to leave it untyped if\\
desired.\\
"""
function union_intercept(expr::Expr)
    (expr, call, name, args) = parse_expr(expr)
    arg_syms = [Expr(:quote, arg[1]) for arg in args]
    diffs = Expr(:vect, [:($(arg[1]) <: Node) for arg in args]...)
    branch_expr = Expr(:call, :branch_expr, name, Expr(:vect, arg_syms...), diffs)
    return Expr(:macrocall, Symbol(:@generated),
        Expr(:function, union_sig(expr), Expr(:return, branch_expr)))
end


function edge_intercept(expr::Expr)

    (expr, call, name, args) = parse_expr(expr)
    tuple_type = to_tuple_type(expr)
    tuple_type_union = to_tuple_type(union_sig(expr))
    arg_syms = Expr(:vect, [Expr(:quote, arg[1]) for arg in args]...)
    arg_names = Expr(:vect, [arg[1] for arg in args]...)

    body = quote

        # Get the signature corresponding to the actual invocation.
        current_sig = Tuple{typeof($name), $(arg_names)...}

        # Get all methods for which mysig is viable and remove mysig.
        ms = Set{Type{T} where T<:Tuple}()
        for m in methods($name)
            if current_sig <: m.sig && !(m.sig <: $tuple_type_union)
                push!(ms, m.sig)
            end
        end

        # Remove all methods which are strictly less specific than any
        # other method supporting the types in question.
        to_pop = []
        for m in ms
            for n in ms
                if n <: m && m != n
                    to_pop = vcat(to_pop, m)
                    continue
                end
            end
        end
        foreach((m)->pop!(ms, m), to_pop)

        # Invoke the method that's left. 
        # More than 1 => ambiguity. 
        # None => throw an error.
        length(ms) > 1 && error("We have an ambiguity!")
        length(ms) == 0 && error("Unsupported types in method $($name)")
        m = collect(ms)[1]
        dt = change_unionall_body(m, Tuple{collect(Base.unwrap_unionall(m).types[2:end])...})
        return Expr(:call, :invoke, Symbol($name), to_expr(dt), $(arg_syms)...)
    end
    return Expr(:macrocall, Symbol(:@generated), Expr(:function, call, body))
end

"""
    to_expr(u)

Convert a method type signature UnionAll to an expression by exploiting the print
functionality. Have to be careful that this will always parse correctly in different
Julia versions.
"""
function to_expr(u)
    buffer = IOBuffer()
    show(buffer, u)
    return parse(String(take!(buffer)))
end

function union_sig(expr::Expr)
    (expr, call, name, args) = parse_expr(expr)
    syms = [gensym() for arg in args]
    new_typed_args = [:($(args[j][1])::Union{$(syms[j]), AutoGrad2.Node{$(syms[j])}})
                        for j in eachindex(args)]
    new_call = Expr(:call, name, new_typed_args...)
    new_typevars = [:($(syms[j])<:$(args[j][2])) for j in eachindex(args)]
    return expr.head == :where ?
        change_where_call(Expr(:where, new_call, new_typevars...), expr) :
        Expr(:where, new_call, new_typevars...)
end

"""
    parse_expr(expr::Expr)

Extract the constituents of an expression such as\\
    expr = :(foo(x::Real, y::T, z::T) where T<:Real)\\
and return them as a Tuple. In this case, the return would be\\
    (expr, :(foo(x::Real, y::T, z::T)), :foo, [(:x, :Real), (:y, :T), (:z, :T)]).\\
If the expression is of the `:curly` format:\\
    expr = :(foo{T<:Real}(x::Real, y::T, z::T))\\
then it is convered to the new `:where` format.
"""
function parse_expr(expr::Expr)
    (isa(expr.args[1], Expr) && expr.args[1].head == :curly) && (expr = curly_to_where(expr))
    call = call_from_where(expr)
    name, args = call.args[1], [parse_arg(arg) for arg in call.args[2:end]]
    return (expr, call, name, args)
end

parse_arg(arg::Symbol) = (arg, :Any)
parse_arg(arg::Expr) = (arg.args[1], arg.args[2])

"""
    branch_expr(f, args::Vector, diffs::Vector{Bool})

Compute Expr that creates a new Branch object whose Function is `f` with\\
arbitrarily named arguments, the number of which is determined by `diffs`.\\
Assumed that at least one element of `diffs` is true.
"""
branch_expr(f, args::Vector, diffs::Vector{Bool}) =
    Expr(:call, :Branch, :($f), Expr(:tuple, args...), get_tape(args, diffs))

"""
    get_tape(args::Vector{Symbol}, diffs::Vector{Bool})

Determines the first argument of `args` which is a `Node` via `diffs`, and returns an\\
`Expr` which returns its tape at run time. If none of the arguments are Nodes, then an\\
error is thrown. Error also thrown if `args` and `diffs` are not the same length.
"""
function get_tape(args::Vector{Symbol}, diffs::Vector{Bool})
    length(args) != length(diffs) && throw(ArgumentError("length(args) != length(diffs)"))
    for j in eachindex(diffs)
        diffs[j] == true && return :($(args[j]).tape)
    end
    throw(ArgumentError("None of the arguments are Nodes."))
end

"""
    parse_sig(sig::DataType)

Parse the Tuple-based method signature to obtain a Tuple containing just the\\
types of the arguments.
"""
parse_sig(sig::DataType) = Tuple{[_parsetype(tp) for tp in sig.types[2:end]]...}
_parsetype(tp) = tp
_parsetype(tp::TypeVar) = tp.ub

"""
    parse_sig(sig::UnionAll)

Parse the UnionAll-based method signature to obtain a Tuple containing just the\\
types of the arguments. This approach can handle parametric types.
"""
parse_sig(sig::UnionAll) = parse_sig(sig.body)

"""
    curly_to_where(expr::Expr)

Convert a`:curly` expression to a `:where` expression.
"""
function curly_to_where(expr::Expr)
    curly = expr.args[1]
    return Expr(:where,
        Expr(:call, curly.args[1], expr.args[2:end]...),
        curly.args[2:end]...)
end

"""
    call_from_where(expr::Expr)

Get the `:call` component of a `where` expression (it is assumed that this particular where)
expression has a `:call` component at the bottom of the recursion).
"""
call_from_where(expr::Expr) = expr.head == :call ? expr : call_from_where(expr.args[1])

"""
    change_where_call(where::Expr, new_call::Expr)

Place a new `:call` expression `new_call` into the `where` expression. It is assumed that
this `where` expression already contains a `:call` expression.
"""
change_where_call(where::Expr, new_call::Expr) =
    where.head == :where ?
        Expr(:where, change_where_call(where.args[1], new_call), where.args[2:end]...) :
        where.head == :call ?
            new_call :
            throw(error("head is neither :call nor :where"))

"""
    to_tuple_type(expr::Expr)

Convert an expression such as\\
    foo(x::Real, y::T, z::T) where T<:Real\\
to a Type-Tuple of the form\\
    Tuple{typeof(foo), Real, T, T} where T<:Real
"""
function to_tuple_type(expr::Expr)
    (expr, call, name, args) = parse_expr(expr)
    tuple_type = Expr(:curly, :Tuple, :(typeof($name)), [arg[2] for arg in args]...)
    return change_where_call(expr, tuple_type)
end

"""
    to_arg_types(m::Method)

Conert a method signature type-Tuple of the form \\
    Tuple{typeof(f), Float64, String}

to a type-tuple containing only the arguments, of the form\\
    Tuple{Float64, String}
"""
to_arg_types(m::Method) =
    change_unionall_body(m.sig, Tuple{Base.unwrap_unionall(m.sig).types[2:end]...})

"""
    change_unionall_body(u::UnionAll, n::Type{T} where T<:Tuple)

Change the body parameter of a `UnionAll`, `u`, with `n`. If !isa(u, UnionAll) then n is
simply returned.
"""
change_unionall_body(old::Type, new::Type{T} where T<:Tuple) = new
change_unionall_body(u::UnionAll, n::Type{T} where T<:Tuple) =
    UnionAll(u.var, change_unionall_body(u.body, n))

""" Used to flag which argument is being specified in x̄. """
struct Arg{N} end

"""
    ∇(::Type{Arg{N}}, f::Function, p, x1, x2, ..., y, ȳ)

To implement a new reverse-mode sensitivity for the `N^{th}` argument of function `f`. p\\
is the output of `preprocess`. `x1`, `x2`,... are the inputs to the function, `y` is its\\
output and `ȳ` the reverse-mode sensitivity of `y`.
"""
function ∇ end

"""
    ∇(x̄, ::Tuple{Arg{N}}, f::Function, args...)

Default implementation for in-place update to sensitivity w.r.t. `N^{th}` argument of\\
function `f`. Calls the allocating version of the routine, creating unecessary\\
temporaries, but providing valid behaviour.
"""
∇(x̄, ::Type{Arg{N}}, f::Function, args...) where N = x̄ + ∇(Arg{N}, f, args...)

"""
    preprocess(::Function, args...)

Default implementation of preprocess returns an empty Tuple. Individual sensitivity\\
implementations should add methods specific to their use case. The output is passed\\
in to `∇` as the 3rd or 4th argument in the new-x̄ and update-x̄ cases respectively.
"""
preprocess(::Function, args...) = ()
