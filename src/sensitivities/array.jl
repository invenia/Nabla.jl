import Base: .+, .-, .*, ./, .\,
    sum, sumabs, sumabs2, prod, maximum, minimum, maxabs, minabs

# # All unary functions of scalars can be performed elementwise. This functionality
# # enforces that. May change with Julia v0.6 when elementwise functions become dotty
# # functions.
# let stype = :AbstractArray
#     for (f, x̄, range) in unary_sensitivities
#         @eval @primitive $f{T <: $stype}(x::T) y ȳ $x̄
#     end
# end

# Sensitivities for elementwise versions of binary operators of the form z = x (op) y.
binary_sensitivities_elementwise = [
    # (:.+, :(z̄), :(z̄[n]), :(z̄), :(z̄[n]), (lb, ub), (lb, ub)),s
    # (:.-, :(copy(z̄)),         :(-z̄),              (lb, ub), (lb, ub)),
    # (:.*, :(z̄ .* y),          :(z̄ .* x),          (lb, ub), (lb, ub)),
    # (:./, :(z̄ ./ y),          :(-z̄ .* x ./ y.^2), (lb, ub), (lb, ub)),
    # (:.\, :(-z̄ .* y ./ x.^2), :(z̄ ./ x),          (lb, ub), (lb, ub)),
]
# binary_sensitivities_elementwise = [
#     (:.+,
#         :()
#         :())
# ]

for (f, x̄_sc, x̄_ar, ȳ_sc, ȳ_ar) in binary_sensitivities_elementwise
    eval(compute_sensitivity_method(f,
        [:(T <: AbstractFloat), :(U <: AbstractFloat), :(V <: AbstractFloat)],
        [:x, :y], [:x̄, :ȳ], [:U, :V], :z, :z̄,
        [:(x̄ = z̄), :(ȳ = z̄)], [:(x̄ += z̄), :(ȳ += z̄)]
    ))
    eval(:(function $f{T <: AbstractArray, U <: AbstractFloat, V <: AbstractArray}(
        tape::Tape, z::T, zn::Int, x::U, y::V, xn::Int, yn::Int)
        z̄::T, v = tape[zn], tape.tape
        x̄ = isdefined(v[xn]) ? v[xn]::U : similar(x)
        ȳ = isdefined(v[yn]) ? v[yn]::V : similar(y)
        if xn > 0
            tmp = 0.0
            @simd for n in eachindex(z̄)
                @inbounds tmp += $x̄_ar
            end
            v[xn] = isdefined(v, xn) ? v[xn] + tmp : tmp
        end
        if yn > 0
            if isdefined(v, yn)
                for n in eachindex(z̄)
                    @inbounds ȳ[n] += $ȳ_ar
                end
            else
                for n in eachindex(z̄)
                    @inbounds ȳ[n] = $ȳ_ar
                end
                v[yn] = ȳ
            end
        end
        return nothing
    end))
    eval(:(function $f{T <: AbstractArray, U <: AbstractArray, V <: AbstractFloat}(
        tape::Tape, z::T, zn::Int, x::U, y::V, xn::Int, yn::Int)
        z̄::T, v = tape[zn], tape.tape
        x̄ = isdefined(v[xn]) ? v[xn]::U : similar(x)
        ȳ = isdefined(v[yn]) ? v[yn]::V : similar(y)
        if xn > 0
            if isdefined(v, xn)
                @inbounds for n in eachindex(z̄)
                    x̄[n] += $x̄_ar
                end
            else
                @inbounds for n in eachindex(z̄)
                    x̄[n] = $x̄_ar
                end
                v[xn] = x̄
            end
        end
        if yn > 0
            tmp = 0.0
            @simd for n in eachindex(z̄)
                @inbounds tmp += $ȳ_ar
            end
            v[yn] = isdefined(v, xn) ? v[yn] + tmp : tmp
        end
        return nothing
    end))
    eval(:(function $f{T <: AbstractArray, U <: AbstractArray, V <: AbstractArray}(
        tape::Tape, z::T, zn::Int, x::U, y::V, xn::Int, yn::Int)
        z̄::T, v = tape[zn], tape.tape
        if xn > 0
            if isdefined(v, xn)
                x̄::U = v[xn]
                @inbounds for n in eachindex(z̄)
                    x̄[n] += z̄[n]
                end
            else
                v[xn] = copy(z̄)
            end
        end
        if yn > 0
            if isdefined(v, yn)
                ȳ::V = v[yn]
                @inbounds for n in eachindex(z̄)
                    ȳ[n] += z̄[n]
                end
            else
                v[yn] = copy(z̄)
            end
        end
        return nothing
    end))
    primitive(f, (:(T <: ArrayOrFloat), :(U <: ArrayOrFloat)), (:T, :U), (true, true))
end

# Basic reductions of a single argument.
reduce = [
    (:sum,
        :(x̄ = broadcast!(x->x, similar(x), ȳ)),
        :(broadcast!(+, x̄, x̄, ȳ))),
    (:sumabs,
        :(x̄ = broadcast!((x, ȳ)->sign(x) * ȳ, similar(x), x, ȳ)),
        :(broadcast!((x̄, x, ȳ)->x̄ + sign(x) * ȳ, x̄, x̄, x, ȳ))),
    (:sumabs2,
        :(x̄ = broadcast!((x, ȳ)->2 * x * ȳ, similar(x), x, ȳ)),
        :(broadcast!((x̄, x, y)->x̄ + 2 * x * ȳ, x̄, x̄, x, y))),
    (:prod,
        :(x̄ = broadcast!((x, y, ȳ)->ȳ * y / x, similar(x), x, y, ȳ)),
        :(broadcast!((x̄, x, y, ȳ)-> x̄ + ȳ * y / x, x̄, x̄, x, y, ȳ))),
    (:maximum,
        :(x̄ = broadcast!((x, y, ȳ)->ȳ * (y == x), similar(x), x, y, ȳ)),
        :(broadcast!((x̄, x, y, ȳ)->x̄ + ȳ * (y == x), x̄, x̄, x, y, ȳ))),
    (:minimum,
        :(x̄ = broadcast!((x, y, ȳ)->ȳ * (y == x), similar(x), x, y, ȳ)),
        :(broadcast!((x̄, x, y, ȳ)->x̄ + ȳ * (y == x), x̄, x̄, x, y, ȳ))),
    (:maxabs,
        :(x̄ = broadcast!((x, y)->sign(x) * (y == abs(x)), similar(x), x, y)),
        :(broadcast!((x̄, x, y)->x̄ + sign(x) * (y == abs(x)), x̄, x̄, x, y))),
    (:minabs,
        :(x̄ = broadcast!((x, y)->sign(x) * (y == abs(x)), similar(x), x, y)),
        :(broadcast!((x̄, x, y)->x̄ + sign(x) * (y == abs(x)), x̄, x̄, x, y))),
]

# Each of the reduce operations, except for sumabs2 which doesn't quite fit the same
# format as the other reduce operations without sacrificing some efficiency.
for (f, new_x̄, update_x̄) in reduce

    # Define the single argument sensitivity.
    eval(compute_sensitivity_method(
        f, [:(T <: AbstractArray)], [:x], [:x̄], [:T], [true], :y, :ȳ, [new_x̄], [update_x̄]
    ))
    primitive(f, (:(T <: AbstractArray),), (:T,), (true,))

    # Define the multiple-argument sensitivity.
    eval(compute_sensitivity_method(
        f, [:(T <: AbstractArray)], [:x, :dims], [:x̄, :nothing], [:T, :Any], [true, false],
            :y, :ȳ, [new_x̄, :nothing], [update_x̄, :nothing]
    ))
    primitive(f, (:(T <: AbstractArray),), (:T, :Any), (true, false))
end
