import Base: .+, .-, .*, ./, .\,
             sum, sumabs, sumabs2, prod, maximum, minimum, maxabs, minabs

# All unary functions of scalars can be performed elementwise. This functionality
# enforces that. May change with Julia v0.6 when elementwise functions become dotty
# functions.
for (f, x̄, range) in unary_sensitivities
    @eval @primitive $f{T <: AbstractArray}(x::T) y ȳ $x̄
end

# Sensitivities for elementwise versions of binary operators.
binary_sensitivities_elementwise = [
    (:.+, :(z̄),               :(z̄),               (lb, ub), (lb, ub)),
    (:.-, :(z̄),               :(-z̄),              (lb, ub), (lb, ub)),
    (:.*, :(z̄ .* y),          :(z̄ .* x),          (lb, ub), (lb, ub)),
    (:./, :(z̄ ./ y),          :(-z̄ .* x ./ y.^2), (lb, ub), (lb, ub)),
    (:.\, :(-z̄ .* y ./ x.^2), :(z̄ ./ x),          (lb, ub), (lb, ub)),
]

# Allow elementwise operations for both scalars and arrays - they should work for both.
for (f, x̄, ȳ, xrange, yrange) in binary_sensitivities_elementwise
    @eval @primitive $f{T, V <: Union{AbstractFloat, AbstractArray}}(x::T, y::V) z z̄ $x̄ $ȳ
end

# Basic reductions of a single argument.
reduce = [
    (:sum,     :(ones(x))),
    (:sumabs,  :(sign(x))),
    (:sumabs2, :(2x)),
    (:prod,    :(y ./ x)),
    (:maximum, :(y .== x)),
    (:minimum, :(y .== x)),
    (:maxabs,  :(sign(x) .* (y .== abs(x)))),
    (:minabs,  :(sign(x) .* (y .== abs(x)))),
]

for (f, x̄) in reduce
    @eval @primitive $f{T <: Union{AbstractFloat, AbstractArray}}(x::T) y ȳ ȳ .* $x̄
    @eval @primitive $f{T <: Union{AbstractFloat, AbstractArray}}(x::T, dims) y ȳ ȳ .* $x̄ false
end
