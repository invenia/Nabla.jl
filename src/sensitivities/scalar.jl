import Base: identity, +, -, *, /, \, ^, sqrt, cbrt,
    sin, cos, tan,
    sind, cosd, tand, sinpi, cospi,
    sec, csc, cot,
    secd, cscd, cotd,
    asin, acos, atan,
    asind, acosd, atand,
    sinh, cosh, tanh,
    asinh, acosh, atanh,
    log, log2, log10, log1p,
    exp, exp2, exp10, expm1,
    deg2rad, rad2deg, significand,
    abs2

# atan2, asind, acosd, atand, asec, acsc,
# acot, asecd, acscd, acotd, sech, csch, coth, asech, acsch, acoth,
# sinc, cosc, hypot,
# min, max, minmax, clamp, abs, copysign, sign, flipsign, erf, erfc, erfx
# erfinv, erfcinv, real, imag, reim, conj, angle, cis, gamma, lgamma, lfact, digamma, invdigamma,
# trigamma, polygamma, airy, airyprime, airyaiprime, airybi, airybiprime, airyx, + all of the other special functions.

π180 = π / 180

# Hand code identity to prevent things from screwing up. This may become uneccessary in a
# future iteration of the code generating code.
identity(x::Node) = Branch(identity, (x,), x.tape)
∇(::typeof(identity), tape::Tape, y, ȳ, x::Real, xid::Int) =
    (xid > 0 && !isassigned(tape.tape, xid)) && (tape.tape[xid] = ȳ)

# Definitions for functions of a single argument written as y = f(x).
unary_sensitivities = [
    (-,     :(-ȳ),                       (lb, ub)),
    (sin,   :(ȳ .* cos(x)),               (lb, ub)),
    (cos,   :(-ȳ .* sin(x)),              (lb, ub)),
    (tan,   :(ȳ .* (1 .+ abs2(y))),        (lb, ub)),
    (sind,  :(ȳ .* π180 .* cosd(x)),   (lb, ub)),
    (cosd,  :(-ȳ .* π180 .* sind(x)),  (lb, ub)),
    (tand,  :(ȳ .* π180 .* (1 .+ y.^2)), (lb, ub)),
    (sinpi, :(ȳ .* π .* cospi(x)),         (lb, ub)),
    (cospi, :(-ȳ .* π .* sinpi(x)),        (lb, ub)),
    (cot,   :(-ȳ .* csc(x).^2),            (lb, ub)),
    (sec,   :(ȳ .* y .* tan(x)),           (lb, ub)),
    (csc,   :(-ȳ .* y .* cot(x)),          (lb, ub)),
    (cotd,  :(-ȳ .* π180 .* cscd(x).^2), (lb, ub)),
    (secd,  :(ȳ .* y .* π180 .* tand(x)), (lb, ub)),
    (cscd,  :(-ȳ .* y .* π180 .* cotd(x)),(lb, ub)),
    (acos,  :(-ȳ ./ sqrt(1 .- abs2(x))),  (-1., 1.)),
    (asin,  :(ȳ ./ sqrt(1 .- abs2(x))),    (-1., 1.)),
    (atan,  :(ȳ ./ (1 .+ abs2(x))),        (lb, ub)),
    # (acosd, :(-ȳ * π / (180 * sqrt(1. - abs2(deg2rad(x))))),  (-rad2deg(1.), rad2deg(1.))),
    # (asin,  :(ȳ / sqrt(1 - abs2(x))),    (-1., 1.)),
    # (atan,  :(ȳ / (1 + abs2(x))),        (lb, ub)),
    (sinh,  :(ȳ .* cosh(x)),              (lb, ub)),
    (cosh,  :(ȳ .* sinh(x)),              (lb, ub)),
    (tanh,  :(ȳ ./ cosh(x).^2),            (lb, ub)),
    (acosh, :(ȳ ./ sqrt(abs2(x) .- 1.)),   (1., ub)),
    (asinh, :(ȳ ./ sqrt(1. .+ abs2(x))),   (lb, ub)),
    (atanh, :(ȳ ./ (1. .- abs2(x))),       (-1. ,1.)),
    (log,   :(ȳ ./ x),                    (0., ub)),
    (log2,  :(ȳ ./ (x .* log(2))),         (0., ub)),
    (log10, :(ȳ ./ (x .* log(10))),        (0., ub)),
    (log1p, :(ȳ ./ (1. .+ x)),             (0., ub)),
    (exp,   :(ȳ .* y),                    (lb, ub)),
    (exp2,  :(ȳ .* y .* log(2)),           (lb, ub)),
    (exp10, :(ȳ .* y .* log(10)),          (lb, ub)),
    (expm1, :(ȳ .* (y .+ 1.)),             (lb, ub)),
    (sqrt,  :(ȳ ./ (2 .* y)),              (0., ub)),
    (cbrt,  :(ȳ ./ (3. .* abs2(y))),      (lb, ub)),
    (deg2rad, :(ȳ .* π180),         (lb, ub)),
    (rad2deg, :(ȳ ./ π180),        (lb, ub)),
    (significand, :(ȳ .* 0.5.^exponent(x)), (lb, ub)),
    (abs2, :(2x), (lb, ub)),
]
for (f, x̄, _) in unary_sensitivities
    new_x̄, update_x̄ = :(x̄ = $x̄), :(x̄ += $x̄)
    @eval @sensitivity $(Symbol(f))(x::Real) (x̄, $new_x̄, $update_x̄) y ȳ
end

# Definitions for functions of two arguments written as z = f(x, y).
binary_sensitivities = [
    (+, :(z̄),                   :(z̄),                (lb, ub), (lb, ub)),
    (-, :(z̄),                   :(-z̄),               (lb, ub), (lb, ub)),
    (*, :(z̄ * y),               :(z̄ * x),            (lb, ub), (lb, ub)),
    (/, :(z̄ / y),               :(-z̄ * x / y^2),     (lb, ub), (lb, ub)),
    (\, :(-z̄ * y / x^2),        :(z̄ / x),            (lb, ub), (lb, ub)),
    (^, :(z̄ * y * z / x), :(z̄ * z * log(x)),   (1e-6, ub), (lb, ub)),
]

for (f, x̄, ȳ, range) in binary_sensitivities
    new_x̄, update_x̄ = :(x̄ = $x̄), :(x̄ += $x̄)
    new_ȳ, update_ȳ = :(ȳ = $ȳ), :(ȳ += $ȳ)
    @eval @sensitivity($(Symbol(f))(x::Real, y::Real),
        [(x̄, $new_x̄, $update_x̄), (ȳ, $new_ȳ, $update_ȳ)], z, z̄)
end

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
