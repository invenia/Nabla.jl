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
    push!(body.args, :(import AutoGrad2))
    push!(body.args, :(using AutoGrad2.DiffCore))
    push!(body.args, :(using AutoGrad2.DiffBase))
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
