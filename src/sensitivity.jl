export sensitivity, @sensitivity

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
    return esc(sensitivity(expr, x̄d, y, ȳ, preprocess))
end

""" Deprecated. Will be removed before release. """
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
    if _method_exists(foo, args)
        sig = parsesig(_which(foo, args).sig)
        defaultexpr = Expr(:call, :invokeexpr, name, sig, :args)
        body = [Expr(:(=), :diffs, Expr(:vect, [:($(arg[1]) <: Node) for arg in args]...)),
                Expr(:(=), :args, Expr(:vect, arg_syms...)),
                Expr(:if, :(any(diffs)), :(return $branchexpr), :(return $defaultexpr))]
    else
        body = [Expr(:(=), :diffs, Expr(:vect, [:($(arg[1]) <: Node) for arg in args]...)),
                Expr(:(=), :args, Expr(:vect, arg_syms...)),
                Expr(:return, branchexpr)]
    end

    # Construct generated function definition.
    intercept =  Expr(:macrocall, Symbol(:@generated),
        Expr(:function, call, Expr(:block, body...)))

    # Symbols for the tape and indices into `tape` to get `x̄`.
    tape, x̄id = gensym(), [gensym() for _ in eachindex(x̄)]

    # Construct signature for the reverse-mode sensitivity computations method.
    typedname = Expr(:curly, name, tpars...)
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
    _method_exists(f::Expr, args::Vector)
Parse parametric type info to ensure dispatch is performed correctly. Specifically avoiding
extending Base.method_exists.
"""
function _method_exists(f::Expr, args::Vector)
    new_args = []
    tpar_dict = Dict([parsearg(tpar) for tpar in f.args[2:end]])
    for (j, arg) in enumerate(args)
        haskey(tpar_dict, arg[2]) ?
            push!(new_args, (arg[1], tpar_dict[arg[2]])) :
            push!(new_args, arg)
    end
    return _method_exists(f.args[1], new_args)
end

"""
    _method_exists(f::Symbol, args::Vector)
Determine whether a method exists that can handle the function specified by `f` with
arguments of the types specified in args. Specifically avoiding extending Base.method_exists
"""
_method_exists(f::Symbol, args::Vector) =
    method_exists(eval(f), Tuple{[eval(arg[2]) for arg in args]...})

"""
    parsesig(sig::DataType)
Parse the Tuple-based method signature to obtain a Tuple containing just the
types of the arguments.
"""
parsesig(sig::DataType) = Tuple{sig.types[2:end]...}

"""
    parsesig(sig::UnionAll)
Parse the UnionAll-based method signature to obtain a Tuple containing just the
types of the arguments. This approach can handle parametric types.
"""
parsesig(sig::UnionAll) = error("Not implemented UnionAll-based signature extraction.")
