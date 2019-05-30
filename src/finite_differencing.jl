using FDM

export check_Dv, check_Dv_update, check_errs,
       assert_approx_equal, domain1, domain2, points, in_domain

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
central_5_1 = central_fdm(5, 1; bound=5e8)
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
equivalent to computing `ȳ'(J f)(x) v`, where `(J f)(x)` denotes the Jacobian of `f`
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

    # Randomly initialise `Leaf`s.
    inits = Vector(undef, length(rtape))
    for i = 1:length(rtape)
        if isleaf(tape(y)[i])
            inits[i] = randned_container(unbox(tape(y)[i]))
            rtape[i] = copy(inits[i])
        end
    end

    # Perform the reverse pass.
    ∇f = propagate(tape(y), rtape)

    # Substract the random initialisations.
    for i = 1:length(rtape)
        isleaf(tape(y)[i]) && (∇f[i] -= inits[i])
    end

    return sum(map((x, v)->sum(∇f[x] .* v), x_, v))
end
compute_Dv_update(f, ȳ::∇ArrayOrScalar, x::∇ArrayOrScalar, v::∇ArrayOrScalar) =
    compute_Dv_update(f, ȳ, (x,), (v,))
isleaf(::Leaf) = true
isleaf(::Any) = false

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
    assert_approx_equal(∇x_alloc, ∇x_fin_diff, ε_abs, ε_rel, "<$f> allocated")
    assert_approx_equal(∇x_inplace, ∇x_fin_diff, ε_abs, ε_rel, "<$f> in-place")
    return true
end

"""
    in_domain(f::Function, x::Float64...)

Check whether an input `x` is in a scalar, real function `f`'s domain.
"""
function in_domain(f::Function, x::Float64...)
    try
        y = f(x...)
        return isa(y, Real) && !isnan(y)
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
function domain1(in_domain::Function, measure::Function, points::Vector{T}) where T
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

   # Return nothing if no domain could be found.
   length(connected_sets) == 0 && return

   # Pick the largest domain.
   return connected_sets[argmax(measure.(connected_sets))]
end

function domain1(f::Function)
    set = domain1(x -> in_domain(f, x), x -> maximum(x) - minimum(x), points)
    set === nothing && return
    return (minimum(set), maximum(set))
end

"""
    Slice2

Slice of a Float64 x Float64 domain.
"""
mutable struct Slice2
    x::Float64
    y_range::Union{Tuple{Float64, Float64}, Nothing}
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
    in_domain_slices = domain1(x -> x.y_range !== nothing, measure, slices)
    in_domain_slices === nothing && return

    # Extract the x range of the domain.
    xs = getfield.(in_domain_slices, :x)
    x_range = (minimum(xs), maximum(xs))

    # Extract the y range of the domain.
    y_ranges = getfield.(in_domain_slices, :y_range)
    y_lower, y_upper = maximum(getindex.(y_ranges, 1)), minimum(getindex.(y_ranges, 2))
    y_lower >= y_upper && return
    y_range = (y_lower, y_upper)

    return (x_range, y_range)
end

# `beta`s domain cannot be determined correctly, since `beta(-.2, -.2)` doesn't throw an
# error, strangely enough.
domain2(::typeof(beta)) = ((minimum(points[points .> 0]), maximum(points)),
                           (minimum(points[points .> 0]), maximum(points)))

# Both of these functions are technically defined on the entire real line, but the left
# half is troublesome due to the large number of points at which it isn't defined. As such
# we restrict unit testing to the right-half.
domain1(::typeof(gamma)) = (minimum(points[points .> 0]), maximum(points[points .> 0]))
domain1(::typeof(trigamma)) = (minimum(points[points .> 0]), maximum(points[points .> 0]))
