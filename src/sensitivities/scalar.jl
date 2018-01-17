using SpecialFunctions
using DiffRules: @define_diffrule, diffrule, diffrules, hasdiffrule

# Hand code the identity because it's really fundamental. It doesn't need to generate a new
# node on the computational graph since it does nothing, but it is useful to have it's
# gradient implemented for use in higher-order functions.
import Base.identity
@explicit_intercepts identity Tuple{Any}
@inline ∇(::typeof(identity), ::Type{Arg{1}}, p, y, ȳ, x) = ȳ
@inline ∇(::typeof(identity), ::Type{Arg{1}}, x::Real) = one(x)

# Some derivatives are missing from DiffRules.
# TODO: Make a PR for DiffRules.
@define_diffrule Base.:\(x, y) = :(-$y / $x^2), :(1 / $x)
@define_diffrule Base.:transpose(x) = :(1)

# Ignore functions that have complex ranges. This may change when Nabla supports complex
# numbers.
ignored_fs = [(:SpecialFunctions, :hankelh1),
              (:SpecialFunctions, :hankelh2),
              (:(Base.Math.JuliaLibm), :log1p),
              (:Base, :rem2pi),
              (:Base, :mod),
              (:Base, :atan2),
              (:Base, :rem)]

unary_sensitivities, binary_sensitivities = [], []

for (package, f, arity) in diffrules()
    (package == :NaNMath || (package, f) in ignored_fs) && continue

    @eval import $package.$f
    if arity == 1
        push!(unary_sensitivities, (package, f))
        ∂f∂x = diffrule(package, f, :x)
        @eval @explicit_intercepts $f Tuple{∇Scalar}
        @eval @inline ∇(::typeof($f), ::Type{Arg{1}}, p, y, ȳ, x::∇Scalar) = ȳ * $∂f∂x
        @eval @inline ∇(::typeof($f), ::Type{Arg{1}}, x::∇Scalar) = $∂f∂x
    elseif arity == 2
        push!(binary_sensitivities, (package, f))
        ∂f∂x, ∂f∂y = diffrule(package, f, :x, :y)
        @eval @explicit_intercepts $f Tuple{∇Scalar, ∇Scalar}
        @eval ∇(::typeof($f), ::Type{Arg{1}}, p, z, z̄, x::∇Scalar, y::∇Scalar) = z̄ * $∂f∂x
        @eval ∇(::typeof($f), ::Type{Arg{2}}, p, z, z̄, x::∇Scalar, y::∇Scalar) = z̄ * $∂f∂y
    else
        error("Cannot implement sensitivity for $package.$f: arity $arity not supported.")
    end
end

# Add method to resolve exponentiation ambiguity.
^(n::Node{<:Real}, p::Integer) = invoke(^, Tuple{Node{<:Real}, Real}, n, p)
