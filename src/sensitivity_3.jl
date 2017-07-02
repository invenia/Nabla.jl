export base_include_str, @differentiable

"""
    add_intercept(foo::Symbol, mod::Symbol)

Add an intercept for the function from module `mod` whose name is `foo` such that calls to\\
it from within an `@differentiable` block of code or module will be tracked.
"""
function add_intercept(foo::Symbol, mod::Symbol)
    return quote 
        @generated function $foo(x...)
            is_node = [issubtype(xj, Node) for xj in x]
            fname = Expr(Symbol("."), $mod, QuoteNode(Symbol($foo)))
            return any(is_node) ?
                Expr(:call, :Branch, fname, :x, :(x[$(findfirst(is_node))].tape)) :
                Expr(:call, fname, Expr(Symbol("..."), :x))
        end
    end
end

"""
    @differentiable code

Make a block of code differentiable. See documentation for details.
"""
macro differentiable(name::Symbol, code::Expr)
    return esc(differentiable(name, code))
end

function differentiable(name::Symbol, code::Expr)
    body = Expr(:block)
    push!(body.args, :(using AutoGrad2))
    push!(body.args, :(import Base))
    push!(body.args, :(Base.include_string(base_include_str(Base.names(AutoGrad2)))))
    push!(body.args, code)
    return Expr(:toplevel, Expr(:module, false, name, body))
end

"""
    base_include_str(exclude_names)

Create a string which, when included (via include_string), imports all of the names from\\
Base which are not present in `exclude_names`.
"""
function base_include_str(exclude_names)
    base_include = "import Base: "
    for name in names(Base)
        if !in(name, exclude_names)
            base_include *= String(name) * ", "
        end
    end
    return base_include[1:end-2]
end
