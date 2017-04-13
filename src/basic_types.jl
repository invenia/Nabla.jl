export getzero, getone

@inline dictit(val::Dict, f::Function) = Dict(n => f(val[n]) for n in eachindex(val))

# Define functionality to return a type-appropriate zero / one / random element.
returns_basic = [
    (:AbstractFloat, :(0.0), :(1.0), :(rand() * (ub - lb) + lb)),
    (:AbstractArray, :(zeros(val)), :(ones(val)), :(rand(size(val)) * (ub - lb) + lb)),
    (:(Union{Set, Tuple}), :(map(getzero, val)), :(map(getone, val)), :(map(getrand, val))),
    (:Dict, :(dictit(val, getzero)), :(dictit(val, getone)), :(dictit(val, getrand))),
]

for (dtype, zeroexpr, oneexpr, randexpr) in returns_basic
    @eval @inline getzero(val::$dtype) = $zeroexpr
    @eval @inline getone(val::$dtype) = $oneexpr
    @eval @inline getrand(val::$dtype) = $randexpr
end
