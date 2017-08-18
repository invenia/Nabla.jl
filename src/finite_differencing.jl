export check_Dv, compute_errs

"""
    approximate_Dv(f::Function, ȳ::ArrayOr∇Real, x::ArrayOr∇Real, v::ArrayOr∇Real)

Estimate the directional derivative of `f` at `x` in direction `v`.
"""
function approximate_Dv(f::Function, ȳ::ArrayOr∇Real, x::ArrayOr∇Real, v::ArrayOr∇Real)
    y1, y2 = f.((x - v, x + v))
    length(y1) == length(ȳ) || throw(ArgumentError("length(y1) != length(y)."))
    return sum(ȳ .* (y2 - y1) / 2)
end
function approximate_Dv(f::Function, ȳ::ArrayOr∇Real, x::∇RealArray, v::∇RealArray)
    length(x) == length(v) || throw(ArgumentError("length(x) != length(v)."))
    type_tuple = Tuple{Function, ArrayOr∇Real, ArrayOr∇Real, ArrayOr∇Real}
    return invoke(approximate_Dv, type_tuple, f, ȳ, x, v)
end

"""
    compute_Dv(f::Function, ȳ::ArrayOr∇Real, x::ArrayOr∇Real, v::ArrayOr∇Real)

Compute the directional derivative of `f` at `x` in direction `v` using AD. Use this
result to back-propagate the sensitivity ȳ. If ȳ, x and v are column vectors, then this is
equivalent to computing `ȳ.'(J f)(x) v`, where `(J f)(x)` denotes the Jacobian of `f`
evaluated at `x`. Analogous operations happen for scalars and N-dimensional arrays.
"""
function compute_Dv(f::Function, ȳ::ArrayOr∇Real, x::ArrayOr∇Real, v::ArrayOr∇Real)
    return sum(∇(f(Leaf(Tape(), x)), ȳ)[1] .* v)
end
function compute_Dv(f::Function, ȳ::ArrayOr∇Real, x::∇RealArray, v::∇RealArray)
    length(x) == length(v) || throw(ArgumentError("length(x) != length(v)."))
    type_tuple = Tuple{Function, ArrayOr∇Real, ArrayOr∇Real, ArrayOr∇Real}
    return invoke(approximate_Dv, type_tuple, f, ȳ, x, v)
end

# Compute the absolute and relative errors between x and y respectively.
compute_errs(x, y) = (abs.(x - y), abs.(x - y) ./ (abs.(x) + 1e-12))

"""
    check_ȳDv(f, ȳ::T, x::T, v::T, tol_abs::∇Real, tol_rel::∇Real) where T<:ArrayOr∇Real

Compare the directional derivative of `f` at `x` in the direction `v` multiplied by the
reverse-mode sensitivity ȳ as computed by Nabla against an estimate produced by finite
differencing. Returns a Tuple containing the absolute and relative errors.
"""
check_Dv(f, ȳ::ArrayOr∇Real, x::T, v::T) where T<:ArrayOr∇Real =
    compute_errs(approximate_Dv(f, ȳ, x, v), compute_Dv(f, ȳ, x, v))
