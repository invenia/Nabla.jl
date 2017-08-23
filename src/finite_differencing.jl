export check_Dv, check_Dv_update, check_errs

"""
    approximate_Dv(
        f::Function,
        ȳ::ArrayOr∇Real,
        x::Tuple{Vararg{ArrayOr∇Real}},
        v::Tuple{Vararg{ArrayOr∇Real}},
    )
    approximate_Dv(f::Function, ȳ::ArrayOr∇Real, x::ArrayOr∇Real, v::ArrayOr∇Real)

Estimate the directional derivative of `f` at `x` in direction `v`. If the function has
multiple arguments, `x` and `v` should be `Tuple`s of inputs and directions respectively.
"""
function approximate_Dv(
    f::Function,
    ȳ::ArrayOr∇Real,
    x::Tuple{Vararg{ArrayOr∇Real}},
    v::Tuple{Vararg{ArrayOr∇Real}},
)
    y1, y2 = f(map(-, x, v)...), f(map(+, x, v)...)
    length(y1) == length(ȳ) || throw(ArgumentError("length(y1) != length(y)."))
    return sum(ȳ .* (y2 - y1) / 2)
end
approximate_Dv(f::Function, ȳ::ArrayOr∇Real, x::ArrayOr∇Real, v::ArrayOr∇Real) =
    approximate_Dv(f, ȳ, (x,), (v,))

"""
    compute_Dv(f::Function, ȳ::ArrayOr∇Real, x::ArrayOr∇Real, v::ArrayOr∇Real)

Compute the directional derivative of `f` at `x` in direction `v` using AD. Use this
result to back-propagate the sensitivity ȳ. If ȳ, x and v are column vectors, then this is
equivalent to computing `ȳ.'(J f)(x) v`, where `(J f)(x)` denotes the Jacobian of `f`
evaluated at `x`. Analogous operations happen for scalars and N-dimensional arrays.
"""
function compute_Dv(
    f::Function,
    ȳ::ArrayOr∇Real,
    x::Tuple{Vararg{ArrayOr∇Real}},
    v::Tuple{Vararg{ArrayOr∇Real}},
)
    x_ = Leaf.(Tape(), x)
    ∇f = ∇(f(x_...), ȳ)
    return sum(map((x, v)->sum(∇f[x] .* v), x_, v))
end
compute_Dv(f::Function, ȳ::ArrayOr∇Real, x::ArrayOr∇Real, v::ArrayOr∇Real) =
    compute_Dv(f, ȳ, (x,), (v,))

function compute_Dv_update(
    f::Function,
    ȳ::ArrayOr∇Real,
    x::Tuple{Vararg{ArrayOr∇Real}},
    v::Tuple{Vararg{ArrayOr∇Real}},
)
    x_ = Leaf.(Tape(), x)
    y = f(x_...)
    rtape = reverse_tape(y, ȳ)
    for n in 1:length(rtape) - 1
        rtape[n] = zerod_container(y.tape[n].val)
    end
    ∇f = propagate(y.tape, rtape)
    return sum(map((x, v)->sum(∇f[x] .* v), x_, v))
end
compute_Dv_update(f::Function, ȳ::ArrayOr∇Real, x::ArrayOr∇Real, v::ArrayOr∇Real) =
    compute_Dv_update(f, ȳ, (x,), (v,))

# Compute the absolute and relative errors between x and y respectively.
compute_errs(x, y) = (abs.(x - y), abs.(x - y) ./ (abs.(x) + 1e-12))

"""
    check_Dv(f, ȳ::T, x::T, v::T) where T

Compare the directional derivative of `f` at `x` in the direction `v` multiplied by the
reverse-mode sensitivity ȳ as computed by Nabla against an estimate produced by finite
differencing. Returns a Tuple containing the absolute and relative errors.
"""
check_Dv(f, ȳ::ArrayOr∇Real, x::T, v::T) where T =
    compute_errs(approximate_Dv(f, ȳ, x, v), compute_Dv(f, ȳ, x, v))

"""
    check_Dv_update(f, ȳ::T, x::T, v::T) where T

Compare the directional derivative of `f` at `x` in the direction `v` multiplied by the
reverse-mode sensitivity ȳ as computed by Nabla with a zerod tape, against an estimate
produced by finite differencing. Returns a Tuple containing the absolute and relative
errors.
"""
check_Dv_update(f, ȳ::T, x::T, v::T) where T =
    compute_errs(approximate_Dv(f, ȳ, x, v), compute_Dv_update(f, ȳ, x, v))

"""
    check_errs(f, ȳ::T, x::T, v::T, ϵ_abs::∇Real, ϵ_rel::∇Real)::Bool where T

Check that the difference between finite differencing directional derivative estimation and
RMAD directional derivative computation for function `f` at `x` in direction `v`, for both
allocating and in-place modes, has absolute and relative errors of `ϵ_abs` and `ϵ_rel`
respectively, when scaled by reverse-mode sensitivity `ȳ`. 
"""
function check_errs(f, ȳ::T, x::T, v::T, ϵ_abs::∇Real, ϵ_rel::∇Real)::Bool where T
    δ_abs_alloc, δ_rel_alloc = check_Dv(f, ȳ, x, v)
    δ_abs_inplace, δ_rel_inlpace = check_Dv_update(f, ȳ, x, v)
    return δ_abs_alloc < ϵ_abs && δ_rel_alloc < ϵ_rel &&
           δ_abs_inplace < ϵ_abs && δ_rel_inlpace < ϵ_rel
end
