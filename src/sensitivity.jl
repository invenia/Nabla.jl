export sensitivity, @sensitivity, branchexpr, invokeexpr, preprocess

"""
    @sensitivity expr x̄ y ȳ preprocess=:nothing
Construct a generated method for the function specified in `expr` which intercepts calls
containing `Node` objects, and performs the bookkeeping necessary to perform RMAD.
Also add a method to `∇` which can compute the reverse-mode sensitivities specified in x̄.

`Add example usage here. (It's comparatively simple to use).`
"""
macro sensitivity(expr, x̄, y, ȳ, preprocess=:nothing)
    x̄v = x̄.head == :vect ? [t for t in x̄.args] : [x̄]
    x̄d = []
    for (j, t) in enumerate(x̄v)
        push!(x̄d, ([s for s in t.args]...))
    end
    out = sensitivity(expr, x̄d, y, ȳ, preprocess)
    println(out)
    return esc(out)
end

""" Possibly remove before open sourcing. """
sensitivity(
    expr::Expr,
    x̄::Tuple,
    y::Symbol,
    ȳ::Symbol,
    preprocess::SymOrExpr=:nothing) =
    sensitivity(expr, Vector{Tuple}([x̄]), y, ȳ, preprocess)

""" Constructs the functions specified by @sensitivity. See @sensitivity for details. """
function sensitivity(
    expr::Expr,
    x̄::Vector,
    y::Symbol,
    ȳ::Symbol,
    preprocess::SymOrExpr=:nothing)

    # Format inputs and check that they aren't `Any`.
    expr.head == :call || error("expr is not a function call")
    args_typed = expr.args[2:end]
    foo, args = expr.args[1], [parsearg(arg) for arg in args_typed]
    (name, tpars) = isa(foo, Expr) ? (foo.args[1], foo.args[2:end]) : (foo, [])
    any([arg[2] == :Any && length(x̄[j]) > 0 for (j, arg) in enumerate(args)]) &&
        error("Types of args must not be Any.")

    # Construct the signature for the generated function.
    syms = [gensym() for arg in args]
    tpars = vcat(tpars, [Expr(:(<:), [syms[j], arg[2]]...) for (j, arg) in enumerate(args)])
    node_params = [Expr(:(::), arg[1], Expr(:curly, :Union, syms[j], :(Node{$(syms[j])})))
                   for (j, arg) in enumerate(args)]
    call = Expr(:call, Expr(:curly, name, tpars...), node_params...)

    # Construct the body of the generated function.
    arg_syms = [Expr(:quote, arg[1]) for arg in args]
    branchexpr = Expr(:call, :branchexpr, name, :args, :diffs)
    body = [Expr(:(=), :diffs, Expr(:vect, [:($(arg[1]) <: Node) for arg in args]...)),
            Expr(:(=), :args, Expr(:vect, arg_syms...))]
    try
        sig = parsesig(_which(foo, args).sig)
        defaultexpr = Expr(:call, :invokeexpr, name, sig, :args)
        body = vcat(body, :(return any(diffs) ? $branchexpr : $defaultexpr))
    catch err
        if isa(err, ErrorException)
            body = vcat(body, :(return $branchexpr))
        else
            throw(err)
        end
    end

    # Construct generated function definition.
    intercept =  Expr(:macrocall, Symbol(:@generated),
        Expr(:function, call, Expr(:block, body...)))

    # Symbols for the tape and indices into `tape` to get `x̄`.
    tape, x̄id = gensym(), [gensym() for _ in eachindex(x̄)]

    # Construct signature for the reverse-mode sensitivity computations method.
    typedname = Expr(:curly, name)
    tape_arg = Expr(:(::), tape, :Tape)
    x̄id_typed = [Expr(:(::), a, Int) for a in x̄id]
    ∇call_name = isa(foo, Symbol) ? :∇ : Expr(:curly, :∇, foo.args[2:end]...)
    ∇call = Expr(:call, ∇call_name, :(::typeof($name)), tape_arg, y, ȳ, args_typed..., x̄id_typed...)

    # Construct body for the reverse-mode sensitivity computations method.
    ∇body = Vector{SymOrExpr}()
    preprocess != :nothing && push!(∇body, preprocess)

    # For each argument in `x`, add code to compute the reverse-mode sensitivity, updating
    # the existing value if present, otherwise creating a new value.
    for n in eachindex(args)
        if length(x̄[n]) > 0
            tape_index = :($tape.tape[$(x̄id[n])])
            update_x̄ = Expr(:block, Expr(:(=), x̄[n][1], tape_index), x̄[n][3])
            push!(∇body,
                Expr(:if, :($(x̄id[n]) > 0), Expr(:block,
                    Expr(:if, :(isassigned($tape.tape, $(x̄id[n]))), update_x̄, x̄[n][2]),
                    Expr(:(=), tape_index, x̄[n][1]))))
        end
    end
    push!(∇body, Expr(:return, :nothing))

    # Construct expression to compute rvs mode sensitivities.
    sensitivity = Expr(:macrocall, Symbol(:@inline),
        Expr(:function, ∇call, Expr(:block, ∇body...)))

    return Expr(:block, intercept, sensitivity)
end

parsearg(arg::Symbol) = (arg, :Any)
parsearg(arg::Expr) = (arg.args[1], arg.args[2])

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
    _which(f::Expr, args::Vector)
Parse parametric type info to ensure dispatch is performed correctly. Specifically avoiding
extending Base.which.
"""
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

"""
    _which(f::Symbol, args::Vector)
Determine which method of `f` will be called given the types of `args`. Specifically
avoiding extending Base.which.
"""
_which(f::Symbol, args::Vector) = which(eval(f), Tuple{[eval(arg[2]) for arg in args]...})

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
    getfuncsymbol(n::Int)
Get the Symbol corresponding to the name of the function which computes the reverse-mode
sensitivity of the n^{th} argument of a function. If it's not present, generate one.
"""
function getfuncsymbol(n::Int)
    haskey(_sens_dict, n) || setindex!(_sens_dict, gensym(), n)
    return _sens_dict[n]
end

# """
#     preprocess(::Function, Tuple, Any, Any) = ()
# Default implementation for preproessing. If there is some preprocessing is required for the
# sensitivities of a particular function then additional methods should be added.
# """
# preprocess(::Function, ::Tuple, ::Any, ::Any) = ()

# """
#     x̄(::Type{Arg{N}}, ::Function, x::Tuple, y::Any, ȳ::Any, p::Any) = 0.0
# Default implementation for a sensitivity throws an error, indicating that a sensitivity for
# a particular argument was requested but unavailable. Methods should be added to this
# function to implement sensitivities for particular methods. For example

#     x̄(Arg{1}, *, x::Tuple{T, V}, ::Any, ȳ::Any, ::Any) = x[2] * ȳ
#     x̄(Arg{2}, *, x::Typle{T, V}, ::Any, ȳ::Any, ::Any) = x[1] * ȳ


# """

# """ Used to flag which argument is being specified in x̄. """
# struct Arg{N} end
# struct Update end
# struct New end

# requires_y(::Function, ::Type) = true

# function add_intercepts(f::Function, types::Type)
#     println("Now spit out an @generated function!")
# end

# function add_sensitivity(call::Expr, arg::Int, expr::Expr)

#     eval(:(x̄(::Type{Arg{$arg}}, ::typeof($f), x::$types, y::Any, ȳ::Any) = $expr))
# end



# # Mock up implementation for *.
# add_intercepts(*, Tuple{T, V} where {T<:Real, V<:Real})
# requires_y{T<:Real, V<:Real}(::typeof(*), ::Type{Tuple{T, V}}) = false

# # From the reverse-pass, I can splat stuff in here and it will just work.
# ∇(::Type{Arg{1}}, ::typeof(*), p, x::Real, y::Real, z̄::Real) = y * z̄
# ∇(::Type{Arg{2}}, ::typeof(*), p, x::Real, y::Real, z̄::Real) = x * z̄

# function add_intercepts(expr::Expr)

#     args = getargs(expr)
#     syms = [gensym() for arg in args]



# end



# # Construct the signature for the generated function.
# syms = [gensym() for arg in args]
# tpars = vcat(tpars, [Expr(:(<:), [syms[j], arg[2]]...) for (j, arg) in enumerate(args)])
# node_params = [Expr(:(::), arg[1], Expr(:curly, :Union, syms[j], :(Node{$(syms[j])})))
#                for (j, arg) in enumerate(args)]
# call = Expr(:call, Expr(:curly, name, tpars...), node_params...)

# # Construct the body of the generated function.
# arg_syms = [Expr(:quote, arg[1]) for arg in args]
# branchexpr = Expr(:call, :branchexpr, name, :args, :diffs)
# body = [Expr(:(=), :diffs, Expr(:vect, [:($(arg[1]) <: Node) for arg in args]...)),
#         Expr(:(=), :args, Expr(:vect, arg_syms...))]
# try
#     sig = parsesig(_which(foo, args).sig)
#     defaultexpr = Expr(:call, :invokeexpr, name, sig, :args)
#     body = vcat(body, :(return any(diffs) ? $branchexpr : $defaultexpr))
# catch err
#     if isa(err, ErrorException)
#         body = vcat(body, :(return $branchexpr))
#     else
#         throw(err)
#     end
# end

# # Construct generated function definition.
# intercept =  Expr(:macrocall, Symbol(:@generated),
#     Expr(:function, call, Expr(:block, body...)))




# x̄, ȳ = :((x̄, z̄ * y, z̄ * y)), :((ȳ, z̄ * x, z̄ * x))
# @eval @sensitivity *(x::T, y::V) where {T<:Real, V<:Real} z z̄ [$x̄, $ȳ]






















# """
# Parse a :call expression, returning the function being called, any parametric types and the
# typed arguments (if they have types).
# """
# function parsecall(call::Expr)
#     call.head == :call || error("Expected a :call or :where expression, got $(call.head).")
#     args1 = call.args[1]
#     f = isa(args1, Symbol) ? args1 : args1.args[1]
#     typevars = isa(args1, Symbol) ? :nothing : args1.args[2:end]
#     return (f, typevars, call.args[2:end])
# end

# """
# Parse a where Expression
# """
# function parsewhere(where::Expr)

# end
