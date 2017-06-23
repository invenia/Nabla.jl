export @getgenintercept, branchexpr, invokeexpr, preprocess, ∇, Arg, parseexpr, parsesig, _genintercept

macro getgenintercept()
    out = quote
        function _which(f::Expr, args::Vector)
            new_args = []
            tpar_dict = Dict([parsearg(tpar) for tpar in f.args[2:end]])
            for (j, arg) in enumerate(args)
                haskey(tpar_dict, arg[2]) ?
                    push!(new_args, (arg[1], tpar_dict[arg[2]])) :
                    push!(new_args, arg)
            end
            return _which(f.args[1], new_args)
        end
        _which(f::Symbol, args::Vector) =
            which(eval(f), Tuple{[eval(arg[2]) for arg in args]...})
        function genintercept(expr::Expr)
            try
                (expr, call, name, args) = parseexpr(expr)
                return AutoGrad2._genintercept(expr, parsesig(_which(name, args).sig))
            catch err
                if isa(err, ErrorException)
                    return AutoGrad2._genintercept(expr, :nothing)
                else
                    throw(err)
                end
            end
        end
    end
    return esc(out)
end
@getgenintercept

"""
    _genintercept(expr::Expr)

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
function _genintercept(expr::Expr, sig)

    (expr, call, name, args) = parseexpr(expr)

    # Construct signature for generated function.
    syms = [gensym() for arg in args]
    new_typed_args = [:($(args[j][1])::Union{$(syms[j]), Node{$(syms[j])}}) for j in eachindex(args)]
    new_call = Expr(:call, name, new_typed_args...)
    new_typevars = [:($(syms[j])<:$(args[j][2])) for j in eachindex(args)]
    new_expr = expr.head == :where ?
        changewherecall(Expr(:where, new_call, new_typevars...), expr) :
        Expr(:where, new_call, new_typevars...)

    # Construct the body of the generated function.
    arg_syms = [Expr(:quote, arg[1]) for arg in args]
    branchexpr = Expr(:call, :branchexpr, name, :args, :diffs)
    body = [Expr(:(=), :diffs, Expr(:vect, [:($(arg[1]) <: Node) for arg in args]...)),
            Expr(:(=), :args, Expr(:vect, arg_syms...))]
    defaultexpr = Expr(:call, :invokeexpr, name, sig, :args)
    body = sig == :nothing ?
        vcat(body, :(return $branchexpr)) :
        vcat(body, :(return any(diffs) ? $branchexpr : $defaultexpr))

    # Construct generated function definition.
    return Expr(:macrocall, Symbol(:@generated),
        Expr(:function, new_expr, Expr(:block, body...)))
end

parsearg(arg::Symbol) = (arg, :Any)
parsearg(arg::Expr) = (arg.args[1], arg.args[2])

function parseexpr(expr::Expr)
    # If the old `curly` format is used then convert to the new `where` format.
    (isa(expr.args[1], Expr) && expr.args[1].head == :curly) && (expr = curlytowhere(expr))
    call = callfromwhere(expr)
    name, args = call.args[1], [parsearg(arg) for arg in call.args[2:end]]
    return (expr, call, name, args)
end

"""
    branchexpr(f, args::Vector, diffs::Vector{Bool})

Compute Expr that creates a new Branch object whose Function is `f` with
arbitrarily named arguments, the number of which is determined by `diffs`.
Assumed that at least one element of `diffs` is true.
"""
branchexpr(f, args::Vector, diffs::Vector{Bool}) =
    Expr(:call, :Branch, :($f), Expr(:tuple, args...), gettape(args, diffs))

"""
    invokeexpr(f, types, args::Vector{Symbol})

Generate an expression which invokes a particular method of the function f. The arguments
should be expressions for the arguments, not the arguments themselves.
"""
invokeexpr(f, types, args::Vector{Symbol}) = Expr(:call, :invoke, :($f), :($types), args...)

"""
    gettape(args::Vector{Symbol}, diffs::Vector{Bool})

Determines the first argument of `args` which is a `Node` via `diffs`, and returns an `Expr`
which returns its tape at run time. If none of the arguments are Nodes, then an error is
thrown. Error also thrown if `args` and `diffs` are not the same length.
"""
function gettape(args::Vector{Symbol}, diffs::Vector{Bool})
    length(args) != length(diffs) && throw(ArgumentError("length(args) != length(diffs)"))
    for j in eachindex(diffs)
        diffs[j] == true && return :($(args[j]).tape)
    end
    throw(ArgumentError("None of the arguments are Nodes."))
end

"""
    parsesig(sig::DataType)

Parse the Tuple-based method signature to obtain a Tuple containing just the
types of the arguments.
"""
parsesig(sig::DataType) = Tuple{[_parsetype(tp) for tp in sig.types[2:end]]...}
_parsetype(tp) = tp
_parsetype(tp::TypeVar) = tp.ub

"""
    parsesig(sig::UnionAll)

Parse the UnionAll-based method signature to obtain a Tuple containing just the
types of the arguments. This approach can handle parametric types.
"""
parsesig(sig::UnionAll) = parsesig(sig.body)

"""
    curlytowhere(expr::Expr)

Convert a`:curly` expression to a `:where` expression.
"""
function curlytowhere(expr::Expr)
    curly = expr.args[1]
    return Expr(:where,
        Expr(:call, curly.args[1], expr.args[2:end]...),
        curly.args[2:end]...)
end

"""
    callfromwhere(expr::Expr)

Get the `:call` component of a `where` expression (it is assumed that this particular where)
expression has a `:call` component at the bottom of the recursion).
"""
callfromwhere(expr::Expr) = expr.head == :call ? expr : callfromwhere(expr.args[1])

"""
    changewherecall(where::Expr, new_call::Expr)

Place a new `:call` expression `new_call` into the `where` expression. It is assumed that
this `where` expression already contains a `:call` expression.
"""
changewherecall(where::Expr, new_call::Expr) =
    where.args[1].head == :call ?
        Expr(:where, new_call, where.args[2:end]...) :
        Expr(:where, changewherecall(where.args[1], new_call), where.args[2:end]...)

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
    ∇(::Type{Arg{N}}, f::Function, args...)

Fallback implementation: throws an error if invoked to indicate that an implementation for\\
a particular sensitivity is not available.
"""
∇(::Type{Arg{N}}, f::Function, args...) where N =
    error("No sensitivity implemented for argument $N of function $f.")

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
