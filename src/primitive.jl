"""
A helper macro used to generate new primitives. y = f(x...) and the corresponding
reverse-mode sensitivities are ȳ and x̄.

Inputs:
f - Expression containing the function.
y - Symbol for the output of the function.
ȳ - Symbol for the reverse-mode sensitivities of the output of the function.
x̄ - tuple of expressions to compute the reverse-mode sensitivities w.r.t. each input.
"""
macro primitive(f::Expr, y::Symbol, ȳ::Symbol, x̄...)

    # Parse the function call and determine which arguments are differentiable.
    fname, pairs = parsefunctioncall(f)
    diff_args = collect(map(x->x != false, x̄))

    # Check that every argument has an associated sensitivity.
    n_args, n_grads = length(pairs), length(x̄)
    n_grads == n_args || error("# args ($n_args) not equal to # grads ($n_grads).")

    # Generate methods which accept Nodes.
    function genmethod(diff_args, states, pos)
        if pos > length(states)
            println("Reached depth.")
            println(states)
            # This recursion appears to be working. Just need to code up the generation of
            # the functions now.
        else
            states[pos] = false
            genmethod(diff_args, states, pos + 1)
            if diff_args[pos]
                states[pos] = true
                genmethod(diff_args, states, pos + 1)
            end
        end
    end
    genmethod(diff_args, copy(diff_args), 1)

    # Construct augmented forward-pass function in terms of Unions of types.
    # fcall = Expr(:(=), fname, )




    # eval(Expr(:Symbol, fname, ))
    # sum{T<:Node}(x::T) = Branch{Float64}(sum, (x,))

end


""" Parse the function call f into it's name and arguments + corresponding types. """
function parsefunctioncall(f::Expr)
    f.head == :call || error("f is not a call, it is a $f.head")
    return f.args[1], map(parsearg, f.args[2:end])
end

""" Argument parsing for parsefunctioncall. """
function parsearg(arg::Expr)
    arg.head == :(::) || error("Expected ::, got $(arg.head).")
    return arg.args[1], arg.args[2]
end
function parsearg(arg::Symbol)
    return (isdefined(arg) && isa(eval(arg), DataType)) ?
        error("Expected an argument name, got a DataType.") : (arg, :Any)
end
parsearg(arg) = error("$j th argument of f is neither a Symbol nor an Expr.")


"""
Construct a function call expression. Simply create an empty call expression and push each
argument onto it in turn.

Inputs:
name - the name of the function. Includes any parametric type information.
args - Vector of tuples of symbols defining argument name-type pairs.

Returns:
the corresponding function call expression.
"""
function constructcall(name::Symbol, args::Tuple)
    f = Expr(:call)
    push!(f.args, name)
    for n in 2:length(args)
        push!(f.args, args[n])
    end
    return f
end
