export @differentiable, add_intercept, Arg, add_∇, add_∇!, ∇, preprocess, intercepts,
    @generated, keys, values

# Contains expressions to generate interceptor methods for all functions.
const intercepts = Dict{Symbol, Expr}()

# Contains the expressions to generate ∇ methods for higher-order functions.
const ∇_functionals = Dict{Symbol, Expr}()

"""
    add_intercept(foo::Symbol, mod::Symbol)
    add_intercept(foo::Symbol)

Add an intercept for the function from module `mod` whose name is `foo` such that calls to\\
it from within an `@differentiable` block of code or module will be tracked.
"""
function add_intercept(foo::Symbol, mod::Symbol)
    intercepts[foo] = quote 
        @generated function $(esc(foo))(x...)
            is_node = [issubtype(xj, Node) for xj in x]
            if any(is_node)
                tape = Expr(:call, :getfield, Expr(:ref, :x, findfirst(is_node)), :(:tape)) 
                return Expr(:call, :Branch, $foo, :x, tape)
            else
                return Expr(:call, $foo, Expr(Symbol("..."), :x))
            end
        end
    end
end
add_intercept(foo, mod) = add_intercept(Symbol(foo), Symbol(mod))

"""
    @differentiable code

Make a block of code differentiable. See documentation for details.
"""
macro differentiable(name::Symbol, code::Expr)
    return differentiable(esc(name), code)
end

function differentiable(name, code)
    body = Expr(:block)
    push!(body.args, :(import Base))
    push!(body.args, :(using AutoGrad2))
    push!(body.args, :(import AutoGrad2.∇))
    foreach(ex->push!(body.args, ex.args[2]), values(intercepts))
    foreach(ex->push!(body.args, ex.args[2]), values(∇_functionals))
    push!(body.args, :(Base.include_string(AutoGrad2.base_include_str(keys(intercepts)))))
    # println(AutoGrad2.base_include_str(keys(intercepts)))
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
∇(x̄, f::Function, ::Type{Arg{N}}, args...) where N = x̄ + ∇(Arg{N}, f, args...)

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
preprocess(::Function, args...) = ()

"""
    needs_output(::Function)

Returns a bool determining whether the particular function in question requires access to\\
its output to compute it's gradient. Defaults to true. Useful for making efficient\\
implementations of `mapreduce` and `mapreducedim`.
"""
needs_output(::Function) = true

