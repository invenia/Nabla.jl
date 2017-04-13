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
    (:.+, :(z̄), :(z̄[n]), :(z̄), :(z̄[n]), (lb, ub), (lb, ub)),
    # (:.-, :(copy(z̄)),         :(-z̄),              (lb, ub), (lb, ub)),
    # (:.*, :(z̄ .* y),          :(z̄ .* x),          (lb, ub), (lb, ub)),
    # (:./, :(z̄ ./ y),          :(-z̄ .* x ./ y.^2), (lb, ub), (lb, ub)),
    # (:.\, :(-z̄ .* y ./ x.^2), :(z̄ ./ x),          (lb, ub), (lb, ub)),
]
for (f, x̄_sc, x̄_ar, ȳ_sc, ȳ_ar) in binary_sensitivities_elementwise
    eval(:(
    function $f{T <: AbstractFloat, U <: AbstractFloat, V <: AbstractFloat}(
        tape::Tape, z::T, zn::Int, x::U, y::V, xn::Int, yn::Int)
        z̄::T, v = tape[zn], tape.tape
        xn > 0 && isdefined(v, xn) ? tape[xn] += $(x̄_sc) : tape[xn] = $(x̄_sc)
        yn > 0 && isdefined(v, yn) ? tape[yn] += $(ȳ_sc) : tape[yn] = $(ȳ_sc)
    end))
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
    (:sum,     :(tape[ypos]),              :(ȳ)),
    (:sumabs,  :(sign(x) * ȳ),             :(sign(x[n]) * ȳ)),
    (:prod,    :(ȳ * y / x),               :(ȳ * y / x[n])),
    (:maximum, :(ȳ * (y == x)),            :(ȳ * (y == x[n]))),
    (:minimum, :(ȳ * (y == x)),            :(ȳ * (y == x[n]))),
    (:maxabs,  :(sign(x) * (y == abs(x))), :(sign(x[n]) * (y == abs(x[n])))),
    (:minabs,  :(sign(x) * (y == abs(x))), :(sign(x[n]) * (y == abs(x[n])))),
]

# Each of the reduce operations, except for sumabs2 which doesn't quite fit the same
# format as the other reduce operations without sacrificing some efficiency.
for (f, ex_fl, ex_ar) in reduce
    eval(:(
        function $f{T <: AbstractFloat, V}(tape::Tape, y::V, ypos::Int, x::T, xpos::Int)
            isdefined(tape, xpos) ? tape[xpos] += $ex_fl : tape[xpos] = $ex_fl
        end
    ))
    eval(:(
        function $f{T <: AbstractArray, V}(tape::Tape, y::V, ypos::Int, x::T, xpos::Int)
            if isdefined(tape.tape, xpos)
                x̄::T, ȳ::V = tape[xpos], tape[ypos]
                @inbounds for n in eachindex(x̄)
                    x̄[n] += $ex_ar
                end
            else
                x̄, ȳ = similar(x)::T, tape[ypos]::V
                @inbounds for n in eachindex(x̄)
                    x̄[n] = $ex_ar
                end
                tape[xpos] = x̄
            end
            return nothing
        end
    ))
    primitive(f, (:(T <: ArrayOrFloat),), (:T,), (true,))
end

# sumabs2.
function sumabs2{T <: AbstractFloat, V}(tape::Tape, y::V, ypos::Int, x::T, xpos::Int)
    isdefined(tape, xpos) ? tape[xpos] += 2x * ȳ : tape[xpos] = 2x * ȳ
end
function sumabs2{T <: AbstractArray, V}(tape::Tape, y::V, ypos::Int, x::T, xpos::Int)
    if isdefined(tape.tape, xpos)
        x̄, ȳ = tape[xpos]::T, tape[ypos]::V
        ȳ2 = 2ȳ
        @inbounds for n in eachindex(x̄)
            x̄[n] += x[n] * ȳ2
        end
    else
        x̄, ȳ = similar(x), tape[ypos]::V
        ȳ2 = 2ȳ
        @inbounds for n in eachindex(x̄)
            x̄[n] = x[n] * ȳ2
        end
        tape[xpos] = x̄
    end
    return nothing
end
primitive(:sumabs2, (:(T <: ArrayOrFloat),), (:T,), (true,))

