export check_Dv, check_Dv_update, check_errs, fdm, forward_fdm, central_fdm, backward_fdm

"""
    approximate_Dv(
        f,
        ȳ::ArrayOr∇Real,
        x::Tuple{Vararg{ArrayOr∇Real}},
            v::Tuple{Vararg{ArrayOr∇Real}},
    )
    approximate_Dv(f::Function, ȳ::ArrayOr∇Real, x::ArrayOr∇Real, v::ArrayOr∇Real)

Estimate the directional derivative of `f` at `x` in direction `v`. If the function has
multiple arguments, `x` and `v` should be `Tuple`s of inputs and directions respectively.
"""
function approximate_Dv(
    f,
    ȳ::ArrayOr∇Real,
    x::Tuple{Vararg{ArrayOr∇Real}},
    v::Tuple{Vararg{ArrayOr∇Real}},
)
    y1, y2 = f(map(-, x, v)...), f(map(+, x, v)...)
    length(y1) == length(ȳ) || throw(ArgumentError("length(y1) != length(y)."))
    return sum(ȳ .* (y2 - y1) / 2)
end
approximate_Dv(f, ȳ::ArrayOr∇Real, x::ArrayOr∇Real, v::ArrayOr∇Real) =
    approximate_Dv(f, ȳ, (x,), (v,))

"""
    compute_Dv(
        f::Function,
        ȳ::ArrayOr∇Real,
        x::Tuple{Vararg{ArrayOr∇Real}},
        v::Tuple{Vararg{ArrayOr∇Real}},
    )

Compute the directional derivative of `f` at `x` in direction `v` using AD. Use this
result to back-propagate the sensitivity ȳ. If ȳ, x and v are column vectors, then this is
equivalent to computing `ȳ.'(J f)(x) v`, where `(J f)(x)` denotes the Jacobian of `f`
evaluated at `x`. Analogous operations happen for scalars and N-dimensional arrays.
"""
function compute_Dv(
    f,
    ȳ::ArrayOr∇Real,
    x::Tuple{Vararg{ArrayOr∇Real}},
    v::Tuple{Vararg{ArrayOr∇Real}},
)
    x_ = Leaf.(Tape(), x)
    ∇f = ∇(f(x_...), ȳ)
    return sum(map((x, v)->sum(∇f[x] .* v), x_, v))
end
compute_Dv(f, ȳ::ArrayOr∇Real, x::ArrayOr∇Real, v::ArrayOr∇Real) =
    compute_Dv(f, ȳ, (x,), (v,))

function compute_Dv_update(
    f,
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
compute_Dv_update(f, ȳ::ArrayOr∇Real, x::ArrayOr∇Real, v::ArrayOr∇Real) =
    compute_Dv_update(f, ȳ, (x,), (v,))

"""
    check_errs(f, ȳ::ArrayOr∇Real, x::T, v::T, ϵ_abs::∇Real, ϵ_rel::∇Real)::Bool where T

Check that the difference between finite differencing directional derivative estimation and
RMAD directional derivative computation for function `f` at `x` in direction `v`, for both
allocating and in-place modes, has absolute and relative errors of `ϵ_abs` and `ϵ_rel`
respectively, when scaled by reverse-mode sensitivity `ȳ`.
"""
function check_errs(f, ȳ::ArrayOr∇Real, x::T, v::T, ϵ_abs::∇Real, c_rel::∇Real)::Bool where T
    ∇x_alloc = compute_Dv(f, ȳ, x, v)
    ∇x_inplace = compute_Dv_update(f, ȳ, x, v)
    ∇x_fin_diff = approximate_Dv(f, ȳ, x, v)

    checks_pass = check_tol(∇x_alloc, ∇x_fin_diff, ϵ_abs, c_rel) &&
                  check_tol(∇x_inplace, ∇x_fin_diff, ϵ_abs, c_rel)
    checks_pass || println("f is $f, ∇x_alloc is $∇x_alloc, ∇x_inplace is $∇x_inplace and ∇x_fin_diff is $∇x_fin_diff")
    return checks_pass
end

check_tol(x, y, ϵ_abs, c_rel) =
    abs.(x - y) .< max(c_rel * eps(c_rel) .* max.(abs.(x), abs.(y)), ϵ_abs)

"""
    FDMReport

Details of a finite-difference method to estimate a derivative. Instances of `FDMReport`
`Base.show` nicely.

# Fields
- `p::Int`: Order of the method.
- `q::Int`: Order of the derivative that is estimated.
- `grid::Vector{<:Real}`: Relative spacing of samples of `f` that are used by the method.
- `coefs::Vector{<:Real}`: Weights of the samples of `f`.
- `ε::Real`: Absolute roundoff error of the function evaluations.
- `M::Real`: Assumed upper bound of `f` and all its derivatives at `x`.
- `ĥ::Real`: Step size.
- `err::Real`: Estimated absolute accuracy.
"""
struct FDMReport
    p::Int
    q::Int
    grid::Vector{<:Real}
    coefs::Vector{<:Real}
    ε::Real
    M::Real
    ĥ::Real
    acc::Real
end
function Base.show(io::IO, x::FDMReport)
    @printf io "FDMReport:\n"
    @printf io "  order of method:       %d\n" x.p
    @printf io "  order of derivative:   %d\n" x.q
    @printf io "  grid:                  %s\n" x.grid
    @printf io "  coefficients:          %s\n" x.coefs
    @printf io "  roundoff error:        %.2e\n" x.ε
    @printf io "  bounds on derivatives: %.2e\n" x.M
    @printf io "  step size:             %.2e\n" x.ĥ
    @printf io "  accuracy:              %.2e\n" x.acc
end

"""
    function fdm(
        grid::Vector{<:Real},
        q::Int;
        ε::Real=1e2 * eps(),
        M::Real=1,
        report::Bool=false
    )

Construct a function `method(f::Function, x::Real, h::Real=ĥ)` that takes in a scalar
function `f`, a point `x` in the domain of `f`, and optionally a step size `h`, and
estimates the `q`'th order derivative of `f` at `x` with a `length(grid)`'th order
finite-difference method.

# Arguments
- `grid::Vector{<:Real}`: Relative spacing of samples of `f` that are used by the method.
    The length of `grid` determines the order of the method.
- `q::Int`: Order of the derivative to estimate. `q` must be strictly less than the order
    of the method.

# Keywords
- `ε::Real=1e2 * eps()`: Absolute roundoff error of the function evaluations.
- `M::Real=1`: Assumed upper bound of `f` and all its derivatives at `x`.
- `report::Bool=false`: Also return an instance of `FDMReport` containing information
    about the method constructed.
"""
function fdm(
    grid::Vector{<:Real},
    q::Int;
    ε::Real=1e2 * eps(),
    M::Real=1,
    report::Bool=false
)
    p = length(grid)  # Order of the method.
    q < p || error("Order of the method must be strictly greater than that of the " *
                   "derivative.")
    p <= 20 || error("Order of the method must be 20 or less.")

    # Compute the coefficients of the FDM.
    C = hcat([grid.^i for i = 0:p - 1]...)'
    x = [i == q + 1 ? factorial(q) : 0 for i = 1:p]
    coefs = C \ x

    # Set the optimal step size by minimising an upper bound on the total error of the
    # estimate.
    C₁ = ε * sum(abs.(coefs))
    C₂ = M * sum(abs.(coefs .* grid.^p)) / factorial(p)
    ĥ = (q / (p - q) * C₁ / C₂) .^ (1 / p)

    # Compute the expected accuracy.
    acc = ĥ^(-q) * C₁ + ĥ^(p - q) * C₂

    # Construct the FDM.
    method(f::Function, x::Real, h::Real=ĥ) = sum(coefs .* f.(x .+ h .* grid)) / h^q

    # If asked for, also return information.
    if report
        return method, FDMReport(p, q, grid, coefs, ε, M, ĥ, acc)
    else
        return method
    end
end
fdm(grid::UnitRange{Int}, args...; kws...) = fdm(Array(grid), args...; kws...)

"""
    backward_fdm(p::Int, ...)
    forward_fdm(p::Int, ...)
    central_fdm(p::Int, ...)

Construct a backward, forward, or central finite-difference method of order `p`. See `fdm`
for further details.

# Arguments
- `p::Int`: Order of the method.

Further takes, in the following order, the arguments `q`, `ε`, `M`, and `report` from `fdm`.
"""
backward_fdm(p::Int, args...; kws...) = fdm(1 - p:0, args...; kws...)
forward_fdm(p::Int, args...; kws...) = fdm(0:p - 1, args...; kws...)
function central_fdm(p::Int, args...; kws...)
    return isodd(p) ? fdm(Int(-(p - 1) / 2):Int((p - 1) / 2), args...; kws...) :
                      fdm([Int(-p/2):-1; 1:Int(p/2)], args...; kws...)
end
