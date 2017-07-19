import Base.include_string
export @differentiable

"""
    @differentiable name code

Make a block of code differentiable. See documentation for details.
"""
macro differentiable(name::Symbol, code::Expr)
    return differentiable(esc(name), code)
end

function differentiable(name, code)
    body = Expr(:block)
    push!(body.args, :(import Base))
    push!(body.args, :(import Nabla))
    push!(body.args, :(using Nabla.DiffCore))
    push!(body.args, :(using Nabla.DiffBase))
    push!(body.args, :(include_string(base_include_str(DiffCore.intercept_names))))
    foreach(arg->push!(body.args, esc(arg)), code.args)
    return Expr(:toplevel, Expr(:module, false, name, body))
end

"""
    base_include_str(exclude_names)

Create a string which, when included (via include_string), imports all of the names from\\
Base which are not present in `exclude_names`.
"""
function base_include_str(exclude_names)
    base_include = "import Base: "
    foreach(x->!in(x, exclude_names) && (base_include *= String(x) * ", "), names(Base))
    return base_include[1:end-2]
end

# """
#     @accepts_nodes func_def

# Ensures that each argument of the function which will be defined by `func_def` can be
# wrapped by a `Node`.
# """
# macro accepts_nodes(code::Expr)
#     head, args = code.head, code.args
#     if head == :function
#         !isa(args[1].head, Expr) && throw(error("Invalid function definition. ($code)"))
#         if args[1].head == :call || args[1].head == :where
            
#         elseif args[1].head == :tuple
            
#         else
#             throw(error("Unrecognised function definition."))
#         end

#     elseif head == Symbol("=") && isa(args[1], Expr) && args[1].head == :call

#     elseif 
# end

# Ways to define a function:

# Not anonymous:
# function foo(x)
#     # code
# end
# foo(x) = x

# Anonymous.
# function(x, y)
#     # code
# end
# (x, y)->code



