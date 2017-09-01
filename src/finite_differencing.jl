export check_Dv, check_Dv_update, check_errs

"""
    approximate_Dv(
        f,
        ȳ::∇ArrayOrScalar,
        x::Tuple{Vararg{∇ArrayOrScalar}},
            v::Tuple{Vararg{∇ArrayOrScalar}},
    )
    approximate_Dv(f::Function, ȳ::∇ArrayOrScalar, x::∇ArrayOrScalar, v::∇ArrayOrScalar)

Estimate the directional derivative of `f` at `x` in direction `v`. If the function has
multiple arguments, `x` and `v` should be `Tuple`s of inputs and directions respectively.
"""
function approximate_Dv(
    f,
    ȳ::∇ArrayOrScalar,
    x::Tuple{Vararg{∇ArrayOrScalar}},
    v::Tuple{Vararg{∇ArrayOrScalar}},
)
    y1, y2 = f(map(-, x, v)...), f(map(+, x, v)...)
    length(y1) == length(ȳ) || throw(ArgumentError("length(y1) != length(y)."))
    return sum(ȳ .* (y2 - y1) / 2)
end
approximate_Dv(f, ȳ::∇ArrayOrScalar, x::∇ArrayOrScalar, v::∇ArrayOrScalar) =
    approximate_Dv(f, ȳ, (x,), (v,))

"""
    compute_Dv(
        f::Function,
        ȳ::∇ArrayOrScalar,
        x::Tuple{Vararg{∇ArrayOrScalar}},
        v::Tuple{Vararg{∇ArrayOrScalar}},
    )

Compute the directional derivative of `f` at `x` in direction `v` using AD. Use this
result to back-propagate the sensitivity ȳ. If ȳ, x and v are column vectors, then this is
equivalent to computing `ȳ.'(J f)(x) v`, where `(J f)(x)` denotes the Jacobian of `f`
evaluated at `x`. Analogous operations happen for scalars and N-dimensional arrays.
"""
function compute_Dv(
    f,
    ȳ::∇ArrayOrScalar,
    x::Tuple{Vararg{∇ArrayOrScalar}},
    v::Tuple{Vararg{∇ArrayOrScalar}},
)
    x_ = Leaf.(Tape(), x)
    ∇f = ∇(f(x_...), ȳ)
    return sum(map((x, v)->sum(∇f[x] .* v), x_, v))
end
compute_Dv(f, ȳ::∇ArrayOrScalar, x::∇ArrayOrScalar, v::∇ArrayOrScalar) =
    compute_Dv(f, ȳ, (x,), (v,))

function compute_Dv_update(
    f,
    ȳ::∇ArrayOrScalar,
    x::Tuple{Vararg{∇ArrayOrScalar}},
    v::Tuple{Vararg{∇ArrayOrScalar}},
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
compute_Dv_update(f, ȳ::∇ArrayOrScalar, x::∇ArrayOrScalar, v::∇ArrayOrScalar) =
    compute_Dv_update(f, ȳ, (x,), (v,))

"""
    check_errs(
        f,
        ȳ::∇ArrayOrScalar,
        x::T,
        v::T,
        ϵ_abs::∇Scalar,
        c_rel::∇Scalar
    )::Bool where T

Check that the difference between finite differencing directional derivative estimation and
RMAD directional derivative computation for function `f` at `x` in direction `v`, for both
allocating and in-place modes, has absolute and relative errors of `ϵ_abs` and `c_rel`
respectively, when scaled by reverse-mode sensitivity `ȳ`.
"""
function check_errs(
    f,
    ȳ::∇ArrayOrScalar,
    x::T,
    v::T,
    ϵ_abs::∇Scalar,
    c_rel::∇Scalar
)::Bool where T
    ∇x_alloc = compute_Dv(f, ȳ, x, v)
    ∇x_inplace = compute_Dv_update(f, ȳ, x, v)
    ∇x_fin_diff = approximate_Dv(f, ȳ, x, v)

    checks_pass = check_tol(∇x_alloc, ∇x_fin_diff, ϵ_abs, c_rel) &&
                  check_tol(∇x_inplace, ∇x_fin_diff, ϵ_abs, c_rel)
    checks_pass || println("f is $f, ∇x_alloc is $∇x_alloc, ∇x_inplace is $∇x_inplace " *
                           "and ∇x_fin_diff is $∇x_fin_diff")
    return checks_pass
end

check_tol(x, y, ϵ_abs, c_rel) =
    abs.(x - y) .< max(c_rel * eps(c_rel) .* max.(abs.(x), abs.(y)), ϵ_abs)
