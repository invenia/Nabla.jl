export check_Dv, check_Dv_update, check_errs, fdm, forward_fdm, central_fdm, backward_fdm,
       check_approx_equal, domain1, domain2, points, in_domain

"""
    approximate_Dv(
        f,
        ȳ::∇ArrayOrScalar,
        x::Tuple{Vararg{∇ArrayOrScalar}},
        v::Tuple{Vararg{∇ArrayOrScalar}}
    )
    approximate_Dv(f::Function, ȳ::∇ArrayOrScalar, x::∇ArrayOrScalar, v::∇ArrayOrScalar)

Estimate the directional derivative of `f` at `x` in direction `v`. If the function has
multiple arguments, `x` and `v` should be `Tuple`s of inputs and directions respectively.
"""
function approximate_Dv(
    f,
    ȳ::∇ArrayOrScalar,
    x::Tuple{Vararg{∇ArrayOrScalar}},
    v::Tuple{Vararg{∇ArrayOrScalar}}
)
    return central_5_1(ε -> sum(ȳ .* f((x .+ ε .* v)...)))
end
approximate_Dv(f, ȳ::∇ArrayOrScalar, x::∇ArrayOrScalar, v::∇ArrayOrScalar) =
    approximate_Dv(f, ȳ, (x,), (v,))

"""
    compute_Dv(
        f,
        ȳ::∇ArrayOrScalar,
        x::Tuple{Vararg{∇ArrayOrScalar}},
        v::Tuple{Vararg{∇ArrayOrScalar}}
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
    v::Tuple{Vararg{∇ArrayOrScalar}}
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
    v::Tuple{Vararg{∇ArrayOrScalar}}
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
        ε_abs::∇Scalar=1e-10,
        ε_rel::∇Scalar=1e-7
    )::Bool where T

Check that the difference between finite differencing directional derivative estimation and
RMAD directional derivative computation for function `f` at `x` in direction `v`, for both
allocating and in-place modes, has absolute and relative errors of `ε_abs` and `ε_rel`
respectively, when scaled by reverse-mode sensitivity `ȳ`.
"""
function check_errs(
    f,
    ȳ::∇ArrayOrScalar,
    x::T,
    v::T,
    ε_abs::∇Scalar=1e-10,
    ε_rel::∇Scalar=1e-7
)::Bool where T
    ∇x_alloc = compute_Dv(f, ȳ, x, v)
    ∇x_inplace = compute_Dv_update(f, ȳ, x, v)
    ∇x_fin_diff = approximate_Dv(f, ȳ, x, v)
    return check_approx_equal("<$f> allocated at $x", ∇x_alloc, ∇x_fin_diff, ε_abs, ε_rel) &
           check_approx_equal("<$f> in-place at $x", ∇x_inplace, ∇x_fin_diff, ε_abs, ε_rel)
end

function check_approx_equal(desc, x, y, ε_abs, ε_rel)
    if abs(x - y) >= ε_abs + ε_rel * (abs(x) + abs(y))
        msg = "\"$desc\": large deviation from reference:\n" *
              "  relative error:  $(@sprintf "%.3e" abs(x - y) / (abs(x) + abs(y)))\n" *
              "    tolerance:     $(@sprintf "%.3e" ε_rel)\n" *
              "  absolute error:  $(@sprintf "%.3e" abs(x - y))\n" *
              "    tolerance:     $(@sprintf "%.3e" ε_abs)\n"
        throw(ErrorException(msg))
    end
    return true
end

"""
    FDMReport

Details of a finite-difference method to estimate a derivative. Instances of `FDMReport`
`Base.show` nicely.

# Fields
- `p::Int`: Order of the method.
- `q::Int`: Order of the derivative that is estimated.
- `grid::Vector{<:∇Scalar}`: Relative spacing of samples of `f` that are used by the method.
- `coefs::Vector{<:∇Scalar}`: Weights of the samples of `f`.
- `ε::∇Scalar`: Absolute roundoff error of the function evaluations.
- `M::∇Scalar`: Assumed upper bound of `f` and all its derivatives at `x`.
- `ĥ::∇Scalar`: Step size.
- `err::∇Scalar`: Estimated absolute accuracy.
"""
struct FDMReport
    p::Int
    q::Int
    grid::Vector{<:∇Scalar}
    coefs::Vector{<:∇Scalar}
    ε::∇Scalar
    M::∇Scalar
    ĥ::∇Scalar
    acc::∇Scalar
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
        grid::Vector{<:∇Scalar},
        q::Int;
        ε::∇Scalar=eps(),
        M::∇Scalar=5e8,
        report::Bool=false
    )

Construct a function `method(f::Function, x::∇Scalar, h::∇Scalar=ĥ)` that takes in a
function `f`, a point `x` in the domain of `f`, and optionally a step size `h`, and
estimates the `q`'th order derivative of `f` at `x` with a `length(grid)`'th order
finite-difference method.

# Arguments
- `grid::Vector{<:∇Scalar}`: Relative spacing of samples of `f` that are used by the method.
    The length of `grid` determines the order of the method.
- `q::Int`: Order of the derivative to estimate. `q` must be strictly less than the order
    of the method.

# Keywords
- `ε::∇Scalar=eps()`: Absolute roundoff error of the function evaluations.
- `M::∇Scalar=5e8`: Assumed upper bound of `f` and all its derivatives at `x`.
- `report::Bool=false`: Also return an instance of `FDMReport` containing information
    about the method constructed.
"""
function fdm(
    grid::Vector{<:∇Scalar},
    q::Int;
    ε::∇Scalar=eps(),
    M::∇Scalar=5e8,
    report::Bool=false
)
    p = length(grid)  # Order of the method.
    q < p || throw(ArgumentError("Order of the method must be strictly greater than that " *
                                 "of the derivative."))

    # Check whether the method can be computed.
    try
        factorial(p)
    catch e
        isa(e, OverflowError) && throw(ArgumentError("Order of the method is too large " *
                                                     "to be computed."))
    end

    # Compute the coefficients of the FDM.
    C = hcat([grid.^i for i = 0:p - 1]...)'
    x = [i == q + 1 ? factorial(q) : 0 for i = 1:p]
    coefs = C \ x

    # Set the step size optimally by minimising an upper bound on the total error of the
    # estimate.
    C₁ = ε * sum(abs.(coefs))
    C₂ = M * sum(abs.(coefs .* grid.^p)) / factorial(p)
    ĥ = (q / (p - q) * C₁ / C₂) .^ (1 / p)

    # Compute the accuracy of the method.
    acc = ĥ^(-q) * C₁ + ĥ^(p - q) * C₂

    # Construct the FDM.
    method(f::Function, x::∇Scalar=0, h::∇Scalar=ĥ) = sum(coefs .* f.(x .+ h .* grid)) / h^q

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

# Precompute some FDMs.
central_3_1 = central_fdm(3, 1)
central_5_1 = central_fdm(5, 1)
central_7_1 = central_fdm(7, 1)

"""
    in_domain(f::Function, x::Float64...)

Check whether an input `x` is in a scalar, real function `f`'s domain.
"""
function in_domain(f::Function, x::Float64...)
    try
        y = f(x...)
        return issubtype(typeof(y), Real) && !isnan(y)
    catch err
        return isa(err, DomainError) ? false : throw(err)
    end
end



# Test points that are used to determine functions's domains.
points = [-π + .1, -.5π + .1, -.9, -.1, .1, .9, .5π - .1, π - .1]

"""
    domain1{T}(in_domain::Function, measure::Function, points::Vector{T})
    domain1(f::Function)

Attempt to find a domain for a unary, scalar function `f`.

# Arguments
- `in_domain::Function`: Function that takes a single argument `x` and returns whether `x`
    argument is in `f`'s domain.
- `measure::Function`: Function that measures the size of a set of points for `f`.
- `points::Vector{T}`: Ordered set of test points to construct the domain from.
"""
function domain1{T}(in_domain::Function, measure::Function, points::Vector{T})
   # Find the connected sets of points that are in f's domain.
   connected_sets, set = Vector{Vector{T}}(), Vector{T}()
   for x in points
       if in_domain(x)
           push!(set, x)
       else
           if length(set) > 0
               push!(connected_sets, set)
               set = Vector{T}()
           end
       end
   end

   # Add the possibly yet unadded set.
   length(set) > 0 && push!(connected_sets, set)

   # Return null if no domain could be found.
   length(connected_sets) == 0 && return Nullable{Vector{T}}()

   # Pick the largest domain.
   return Nullable(connected_sets[indmax(measure.(connected_sets))])
end

function domain1(f::Function)
    set = domain1(x -> in_domain(f, x), x -> maximum(x) - minimum(x), points)
    isnull(set) && return Nullable{NTuple{2, Float64}}()
    return Nullable((minimum(get(set)), maximum(get(set))))
end

"""
    Slice2

Slice of a Float64 x Float64 domain.
"""
type Slice2
    x::Float64
    y_range::Nullable{Tuple{Float64, Float64}}
end

"""
    domain2(f::Function)

Attempt to find a rectangular domain for a binary, scalar function `f`.
"""
function domain2(f::Function)
    # Construct slices for all x in points.
    slices = Slice2.(points, [domain1(y -> f(x, y)) for x in points])

    # Extract a set of in-domain slices.
    measure = x -> maximum(getfield.(x, :x)) - minimum(getfield.(x, :x))
    in_domain_slices = domain1(x -> !isnull(x.y_range), measure, slices)
    isnull(in_domain_slices) && return Nullable{NTuple{2, NTuple{2, Float64}}}()

    # Extract the x range of the domain.
    xs = getfield.(get(in_domain_slices), :x)
    x_range = (minimum(xs), maximum(xs))

    # Extract the y range of the domain.
    y_ranges = get.(getfield.(get(in_domain_slices), :y_range))
    y_lower, y_upper = maximum(getindex.(y_ranges, 1)), minimum(getindex.(y_ranges, 2))
    y_lower >= y_upper && return Nullable{NTuple{2, NTuple{2, Float64}}}()
    y_range = (y_lower, y_upper)

    return Nullable((x_range, y_range))
end

# `beta`s domain cannot be determined correctly, since `beta(-.2, -.2)` doesn't throw an
# error, strangely enough.
domain2(::typeof(beta)) = Nullable(((minimum(points[points .> 0]), maximum(points)),
                                    (minimum(points[points .> 0]), maximum(points))))

# Both of these functions are technically defined on the entire real line, but the left
# half is troublesome due to the large number of points at which it isn't defined. As such
# we restrict unit testing to the right-half.
domain1(::typeof(gamma)) = Nullable((minimum(points[points .> 0]), maximum(points[points .> 0])))
domain1(::typeof(trigamma)) = Nullable((minimum(points[points .> 0]), maximum(points[points .> 0])))
