export primitive

"""
    primitive(f::Symbol, typepars::Vector, argtypes::Vector, diffs::Vector)
Construct the methods of `f` required to enable it propagate derivatives correctly. `diffs`
indicates the arguments which are differentiable.
"""
function primitive(f::Symbol, typepars::Vector, argtypes::Vector, diffs::Vector)
    function primitive_(states::Vector{Bool}, pos::Int)
        if pos > length(states)
            any(states .== true) && eval(constructboxedfunc(f, typepars, argtypes, states))
        else
            states[pos] = false
            primitive_(states, pos + 1)
            if diffs[pos]
                states[pos] = true
                primitive_(states, pos + 1)
            end
        end
    end
    primitive_(collect(diffs), 1)
end


"""
    constructboxedfunc(f::Symbol, tpars::Vector, argts::Vector, diffs::Vector{Bool})
Construct a method of the Function `f`, with parametric typess `tpars`, arguments with the
types specified by `argts`. Arguments which are expected to be `Node` objects should be
indicated by `true` values in `diffs`.
"""
function constructboxedfunc(f::Symbol, tpars::Vector, argts::Vector, diffs::Vector{Bool})
    Expr(:(=), callexpr(f, tpars, argts, diffs), branchexpr(f, diffs))
end


"""
    callexpr(f::Symbol, typepars::Vector, argtypes::Vector, diffs::Vector{Bool})
Compute Expr which creates a new function call.

Inputs:\\\
`f::Symbol` - Function to call.\\
`typepars::Vector` - Parametric type information.\\
`argtypes::Vector` - the type of the arguments. Can refer to something in `typepars`.\\
`diffs::Vector{Bool}` - the arguments which will be Nodes.
"""
function callexpr(f::Symbol, typepars::Vector, argtypes::Vector, diffs::Vector{Bool})
    ex = Expr(:call, Expr(:curly, f, typepars...))
    for n in eachindex(diffs)
        datatype = diffs[n] ? Expr(:curly, :Node, argtypes[n]) : argtypes[n]
        push!(ex.args, Expr(:(::), Symbol("x", n), datatype))
    end
    return ex
end


"""
    branchexpr(f::Symbol, diffs::Vector{Bool})
Compute Expr which creates a new Branch object whose Function is `f` with
arbitrarily named arguments, the number of which is determined by `diffs`.
Assumed that at least one element of `diffs` is true.
"""
function branchexpr(f::Symbol, diffs::Vector{Bool})
    args = [Symbol("x", n) for n in eachindex(diffs)]
    return Expr(:call, :Branch, f, Expr(:tuple, args...), gettape(args, diffs))
end


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
