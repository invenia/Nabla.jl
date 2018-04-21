using SpecialFunctions
using DiffRules: @define_diffrule, diffrule, diffrules, hasdiffrule

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
    @eval @∇primitive $package.$f

    if arity == 1
        push!(unary_sensitivities, (package, f))
        ∂f∂x = diffrule(package, f, :x)
        @eval @inline ∇(::typeof($f), ::Type{Val{1}}, p, y, ȳ, x::∇Scalar) = ȳ * $∂f∂x
        @eval @inline ∇(::typeof($f), ::Type{Val{1}}, x::∇Scalar) = $∂f∂x
    elseif arity == 2
        push!(binary_sensitivities, (package, f))
        ∂f∂x, ∂f∂y = diffrule(package, f, :x, :y)
        @eval ∇(::typeof($f), ::Type{Val{1}}, p, z, z̄, x::∇Scalar, y::∇Scalar) = z̄ * $∂f∂x
        @eval ∇(::typeof($f), ::Type{Val{2}}, p, z, z̄, x::∇Scalar, y::∇Scalar) = z̄ * $∂f∂y
    else
        error("Cannot implement sensitivity for $package.$f: arity $arity not supported.")
    end
end
