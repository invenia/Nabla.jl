
# export @primitive
# typealias ExSym Union{Expr, Symbol}

# """
# A helper macro used to generate new primitives. y = f(x...) and the corresponding
# reverse-mode sensitivities are ȳ and x̄.

# Inputs:
# f - Expression containing the function.
# y - Symbol for the output of the function.
# ȳ - Symbol for the reverse-mode sensitivities of the output of the function.
# x̄ - tuple of expressions to compute the reverse-mode sensitivities w.r.t. each input.
# """
# macro primitive(f::Expr, y::Symbol, ȳ::Symbol, x̄...)

#     # Parse the function call and determine which arguments are differentiable.
#     fname, pairs = parsefunctioncall(f)
#     diff_args = collect(map(x->x != false, x̄))

#     # Check that every argument has an associated sensitivity.
#     n_args, n_grads = length(pairs), length(x̄)
#     n_grads == n_args || error("# args ($n_args) not equal to # grads ($n_grads).")

#     # Construct methods with boxed arguments.
#     methods = genmethods(diff_args, fname, pairs)

#     # Generate reverse-mode sensitivities method.
#     args = vcat(collect(((:diff, :Node), (y, :Any), (ȳ, :Any))), pairs)
#     fcall = constructcall(fname, args, [false for n in eachindex(args)])
#     sensitivities = Expr(:tuple, x̄...)

#     # Generate and return block of code which defines the functions.
#     return esc(Expr(:block, methods..., Expr(:(=), fcall, sensitivities)))
# end


# """ Parse the function call f into it's name and arguments + corresponding types. """
# function parsefunctioncall(f::Expr)
#     f.head == :call || error("f is not a call, it is a $f.head")
#     return f.args[1], map(parsearg, f.args[2:end])
# end

# """ Argument parsing for parsefunctioncall. """
# function parsearg(arg::Expr)
#     arg.head == :(::) || error("Expected ::, got $(arg.head).")
#     return arg.args[1], arg.args[2]
# end
# function parsearg(arg::Symbol)
#     return (isdefined(arg) && isa(eval(arg), DataType)) ?
#         error("Expected an argument name, got a DataType.") : (arg, :Any)
# end
# parsearg(arg) = error("$j th argument of f is neither a Symbol nor an Expr.")


# """
# Generate each of the methods required for dispatch on variables of type Node.

# Inputs:
# diff_args - Vector of bools indicating which variables to box.
# """
# function genmethods(diff_args::Vector{Bool}, fname::Union{Symbol, Expr}, pairs::Vector)
#     exprs = Vector{Expr}()
#     function genmethods_(diff_args::Vector{Bool}, states::Vector{Bool}, pos::Int)
#         if pos > length(states)
#             if any(states .== true)
#                 push!(exprs, constructboxedfunc(fname, pairs, states))
#             end
#         else
#             states[pos] = false
#             genmethods_(diff_args, states, pos + 1)
#             if diff_args[pos]
#                 states[pos] = true
#                 genmethods_(diff_args, states, pos + 1)
#             end
#         end
#     end
#     genmethods_(diff_args, copy(diff_args), 1)
#     return exprs
# end


# """
# Construct a method which contains some boxed arguments. For example,
#     f{T<:AbstractFloat}(x::Vector{T})
# could be used to construct the boxed version:
#     f{T<:AbstractFloat}(x::Node{Vector{T}}) = Branch(f, (x,))

# Inputs:
# fname - name of the function. Includes parametric type information.
# pairs - vector of tuples of symbol / expression pairs defining name-type combinations.
# box   - vector indicating which variables to box.

# Returns:
# Expr representing the method.
# """
# function constructboxedfunc(fname::ExSym, pairs::Vector, box::Vector{Bool})
#     ex = Expr(:(=))
#     push!(ex.args, constructcall(fname, pairs, box), constructbranch(fname, pairs, box))
#     return ex
# end


# """
# Construct a function call expression. Simply create an empty call expression and push each
# argument onto it in turn.

# Inputs:
# fname - the name of the function. Includes any parametric type information.
# pairs - Vector of tuples of symbols defining argument name-type pairs.
# box   - vector indicating which variables to box.

# Returns:
# the corresponding function call expression.
# """
# function constructcall(fname::ExSym, pairs::Vector, box::Vector{Bool})
#     f = Expr(:call)
#     push!(f.args, fname)
#     for n in 1:length(pairs)
#         datatype = box[n] ? Expr(:curly, :Node, pairs[n][2]) : pairs[n][2]
#         push!(f.args, Expr(:(::), pairs[n][1], datatype))
#     end
#     return f
# end


# """
# Construct the correct Branch object.

# Inputs:
# fname - function name.
# pairs - argument (name, DataType) pairs.

# Returns:
# An expression which can be used to define the function.
# """
# constructbranch(fname::Expr, pairs, box) = constructbranch(fname.args[1], pairs, box)
# function constructbranch(fname::Symbol, pairs::Vector, box::Vector{Bool})
#     args = [a for (a, b) in pairs]
#     return Expr(:call, :Branch, fname, Expr(:tuple, args...), gettape(pairs, box))
# end

# """
# Identify the first of the arguments that will contain a tape object by finding the first
# element whos type is a subtype of Node.

# Inputs:
# pairs - argument value-type pairings.

# Returns:
# An expression to get the tape from one of the arguments.
# """
# function gettape(pairs::Vector, box::Vector{Bool})
#     for j in eachindex(box)
#         if box[j] == true
#             # dump(Expr(:., pairs[j][1], :tape))
#             # return Expr(:., pairs[j][1], :tape)
#             return :($(pairs[j][1]).tape)
#         end
#     end
#     throw(ArgumentError("None of the arguments are Nodes."))
# end
