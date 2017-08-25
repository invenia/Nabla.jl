# Hand code the identity because it's really fundamental. It doesn't need to generate a new
# node on the computational graph since it does nothing, but it is useful to have it's
# gradient implemented for use in higher-order functions.
import Base.identity
@explicit_intercepts identity Tuple{Any} 
@inline ∇(::typeof(identity), ::Type{Arg{1}}, p, y, ȳ, x) = ȳ
@inline ∇(::typeof(identity), ::Type{Arg{1}}, x::Real) = one(x)

_ϵ, lb, ub = 3e-2, -3.0, 3.0
binary_sensitivities = (
    (:+, :(z̄),             :(z̄),              (lb, ub), (lb, ub)),
    (:-, :(z̄),             :(-z̄),             (lb, ub), (lb, ub)),
    (:*, :(z̄ * y),         :(z̄ * x),          (lb, ub), (lb, ub)),
    (:/, :(z̄ / y),         :(-z̄ * x / y^2),   (lb, ub), (lb, ub)),
    (:\, :(-z̄ * y / x^2),  :(z̄ / x),          (lb, ub), (lb, ub)),
    (:^, :(z̄ * y * z / x), :(z̄ * z * log(x)), (_ϵ, ub), (_ϵ, ub)),
)
for (f, x̄, ȳ, r1, r2) in binary_sensitivities
    @eval import Base.$f
    @eval @explicit_intercepts $f Tuple{∇Real, ∇Real}
    @eval ∇(::typeof($f), ::Type{Arg{1}}, p, z, z̄, x::∇Real, y::∇Real) = $x̄
    @eval ∇(::typeof($f), ::Type{Arg{2}}, p, z, z̄, x::∇Real, y::∇Real) = $ȳ
end

# Definitions for functions of a single argument written as y = f(x). The boolean argument
# indicates whether the gradient computation requires access to y or not.
unary_sensitivities = (
    (:-,     :(-1),                           (lb, ub),          false),
    (:sin,   :(1 * cos(x)),                   (lb, ub),          false),
    (:cos,   :(-1 * sin(x)),                  (lb, ub),          false),
    (:tan,   :(1 * (1 + abs2(y))),            (-π/2 + 1e-3, π/2 - 1e-3), true),
    (:sind,  :(1 * (π / 180) * cosd(x)),      (lb, ub),          false),
    (:cosd,  :(-1 * (π / 180) * sind(x)),     (lb, ub),          false),
    (:tand,  :(1 * (π / 180) * (1 + y^2)),    (lb, ub),          true),
    (:sinpi, :(1 * π * cospi(x)),             (lb, ub),          false),
    (:cospi, :(-1 * π * sinpi(x)),            (lb, ub),          false),
    (:cot,   :(-1 * csc(x)^2),                (_ϵ, π - _ϵ),          false),
    (:sec,   :(1 * y * tan(x)),               (-0.5π + _ϵ, 0.5π - _ϵ), true),
    (:csc,   :(-1 * y * cot(x)),              (_ϵ, π - _ϵ),      true),
    (:cotd,  :(-1 * (π / 180) * cscd(x)^2),   (lb, ub),          false),
    (:secd,  :(1 * y * (π / 180) * tand(x)),  (lb, ub),          true),
    # (:cscd,  :(-1 * y * (π / 180) * cotd(x)), (lb, ub),          true),
    (:acos,  :(-1 / sqrt(1 - abs2(x))),       (-1 + _ϵ, 1 - _ϵ), false),
    (:asin,  :(1 / sqrt(1 - abs2(x))),        (-1 + _ϵ, 1 - _ϵ), false),
    (:atan,  :(1 / (1 + abs2(x))),            (lb, ub),          false),
    # (acosd, :(-ȳ * π / (180 * sqrt(1. - abs2(deg2rad(x))))),  (-rad2deg(1.), rad2deg(1.))),
    (:sinh,  :(cosh(x)),                      (lb, ub),          false),
    (:cosh,  :(sinh(x)),                      (lb, ub),          false),
    (:tanh,  :(1 / cosh(x)^2),                (lb, ub),          false),
    (:acosh, :(1 / sqrt(abs2(x) - 1)),        (1 + _ϵ, ub),      false),
    (:asinh, :(1 / sqrt(1 + abs2(x))),        (lb, ub),          false),
    (:atanh, :(1 / (1 - abs2(x))),            (-1 + _ϵ, 1 - _ϵ), false),
    (:log,   :(1 / x),                        (_ϵ, ub),          false),
    (:log2,  :(1 / (x * log(2))),             (_ϵ, ub),          false),
    (:log10, :(1 / (x * log(10))),            (_ϵ, ub),          false),
    (:log1p, :(1 / (1 + x)),                  (_ϵ, ub),          false),
    (:exp,   :(y),                            (lb, ub),          true),
    (:exp2,  :(y * log(2)),                   (lb, ub),          true),
    # (:exp10, :(y * log(10)),                  (-3.0, 3.0),       true),
    (:expm1, :((y + 1)),                      (lb, ub),          true),
    (:sqrt,  :(1 / (2 * y)),                  (_ϵ, ub),          true),
    (:cbrt,  :(1 / (3 * abs2(y))),            (lb, ub),          true),
    (:deg2rad, :(1 * (π / 180)),              (lb, ub),          false),
    (:rad2deg, :(1 / (π / 180)),              (lb, ub),          false),
    # (:significand, :(0.5^exponent(x)),        (lb, ub),          false),
    (:abs2, :(2x),                            (lb, ub),          false),
)

# Create implementations for each scalar function such that they can be used in the
# implementation of sensitivities.
for (f, x̄, _, needs_y) in unary_sensitivities
    @eval import Base.$f
    @eval @explicit_intercepts $f Tuple{∇Real}
    @eval @inline ∇(::typeof($f), ::Type{Arg{1}}, p, y, ȳ, x::∇Real) = ȳ * $x̄

    if needs_y
        @eval @inline ∇(::typeof($f), ::Type{Arg{1}}, x::∇Real, y) = $x̄
    else
        @eval @inline ∇(::typeof($f), ::Type{Arg{1}}, x::∇Real) = $x̄
    end
    @eval needs_output(::typeof($f)) = $needs_y
end

# Add method to resolve exponentiation ambiguity.
^(n::Node{<:Real}, p::Integer) = invoke(^, Tuple{Node{<:Real}, Real}, n, p)

# A collection of unary sensitivites yet to be implemented.

# atan2, asind, acosd, atand, asec, acsc,
# acot, asecd, acscd, acotd, sech, csch, coth, asech, acsch, acoth,
# sinc, cosc, hypot,
# min, max, minmax, clamp, abs, copysign, sign, flipsign, erf, erfc, erfx
# erfinv, erfcinv, real, imag, reim, conj, angle, cis, gamma, lgamma, lfact, digamma, invdigamma,
# trigamma, polygamma, airy, airyprime, airyaiprime, airybi, airybiprime, airyx, + all of the other special functions.

# (The rest of the list of functions mentioned in AutoGrad.jl)
# Other functions defined in julia/base/math.jl
# frexp: returns tuple
# minmax: returns tuple
# modf: returns tuple
# Moved to erf.jl:
# erf: see erf.jl
# erfc: see erf.jl
# Moved to gamma.jl:
# (lgamma, :(digamma(x)), (-Inf,Inf)),
