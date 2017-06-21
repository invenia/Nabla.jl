import Base: .+, .-, .*, ./, .\,
    sum, sumabs, sumabs2, prod, maximum, minimum, maxabs, minabs
import Base.Broadcast.broadcast_shape


"""
    broadcastsum!(f::Function, add::Bool, z, As...)

Broadcast f over As and reduce to z by summing. If add is true, then the result is added to
the current value of z, otherwise it is overwritten.
"""
function broadcastsum!(f::Function, add::Bool, z, As...)
    tmp_shape = map(x->x.stop, broadcast_shape(As...))
    if size(z) != tmp_shape
        tmp = Array(eltype(z), tmp_shape)
        return sum!(z, broadcast!((x...)->f(x...), tmp, As...), init=!add)
    else
        return add ?
            broadcast!((z, x...)->z + f(x...), z, z, As...) :
            broadcast!((x...)->f(x...), z, As...)
    end
end


"""
    broadcastsum(f::Function, add::Bool, z::AbstractArray, As...)

Allocating version of broadcastsum! specialised for Arrays.
"""
function broadcastsum(f::Function, add::Bool, z::AbstractArray, As...)
    return broadcastsum!(f, add, similar(z), As...)
end


"""
    broadcastsum(f::Function, add::Bool, z::Number, As...)

Specialisation of broadcastsum to Number-sized outputs.
"""
function broadcastsum(f::Function, add::Bool, z::Number, As...)
    tmp = Array(eltype(z), map(x->x.stop, broadcast_shape(As...)))
    return sum(broadcast!((x...)->f(x...), tmp, As...)) + add ? z : 0.
end


# Sensitivities for elementwise versions of binary operators of the form z = x (op) y.
binary_sensitivities_elementwise = [
    (:.+,
        :(x̄ = broadcastsum(z̄->z̄, false, x, z̄)),
        :(ȳ = broadcastsum(z̄->z̄, false, y, z̄)),
        :(x̄ = broadcastsum!(z̄->z̄, true, x̄, z̄)),
        :(ȳ = broadcastsum!(z̄->z̄, true, ȳ, z̄)),
        (lb, ub), (lb, ub)),
    (:.-,
        :(x̄ = broadcastsum(z̄->z̄, false, x, z̄)),
        :(ȳ = broadcastsum(z̄->-z̄, false, y, z̄)),
        :(x̄ = broadcastsum!(z̄->z̄, true, x̄, z̄)),
        :(ȳ = broadcastsum!(z̄->-z̄, true, ȳ, z̄)),
        (lb, ub), (lb, ub)),
    (:.*,
        :(x̄ = broadcastsum((y, z̄)->y * z̄, false, x, y, z̄)),
        :(ȳ = broadcastsum((x, z̄)->x * z̄, false, y, x, z̄)),
        :(x̄ = broadcastsum!((y, z̄)->y * z̄, true, x̄, y, z̄)),
        :(ȳ = broadcastsum!((x, z̄)->x * z̄, true, ȳ, x, z̄)),
        (lb, ub), (lb, ub)),
    (:./,
        :(x̄ = broadcastsum((y, z̄)->z̄ / y, false, x, y, z̄)),
        :(ȳ = broadcastsum((x, y, z̄)->-z̄ * x / y^2, false, y, x, y, z̄)),
        :(x̄ = broadcastsum!((y, z̄)->z̄ / y, true, x̄, y, z̄)),
        :(ȳ = broadcastsum!((x, y, z̄)->-z̄ * x / y^2, true, ȳ, x, y, z̄)),
        (lb, ub), (lb, ub)),
    (:.\,
        :(x̄ = broadcastsum((x, y, z̄)-> -z̄ * y / x^2, false, x, x, y, z̄)),
        :(ȳ = broadcastsum((x, z̄)->z̄ / x, false, y, x, z̄)),
        :(x̄ = broadcastsum!((x, y, z̄)-> -z̄ * y / x^2, true, x̄, x, y, z̄)),
        :(ȳ = broadcastsum!((x, z̄)->z̄ / x, true, ȳ, x, z̄)),
        (lb, ub), (lb, ub)),
]

# for (f, new_x̄, new_ȳ, update_x̄, update_ȳ, xr, yr) in binary_sensitivities_elementwise
#     generate_primitive(f, [:(T <: ArrayOrFloat), :(U <: ArrayOrFloat)], [:x, :y], [:x̄, :ȳ],
#         [:T, :U], [true, true], :z, :z̄, [new_x̄, new_ȳ], [update_x̄, update_ȳ])
# end

# Basic reductions of a single argument.
reduce = [
    (sum,
        :(x̄ = broadcast!(x->x, similar(x), ȳ)),
        :(broadcast!(+, x̄, x̄, ȳ))),
    # (sumabs,
    #     :(x̄ = broadcast!((x, ȳ)->sign(x) * ȳ, similar(x), x, ȳ)),
    #     :(broadcast!((x̄, x, ȳ)->x̄ + sign(x) * ȳ, x̄, x̄, x, ȳ))),
    # (sumabs2,
    #     :(x̄ = broadcast!((x, ȳ)->2 * x * ȳ, similar(x), x, ȳ)),
    #     :(broadcast!((x̄, x, y)->x̄ + 2 * x * ȳ, x̄, x̄, x, y))),
    # (prod,
    #     :(x̄ = broadcast!((x, y, ȳ)->ȳ * y / x, similar(x), x, y, ȳ)),
    #     :(broadcast!((x̄, x, y, ȳ)-> x̄ + ȳ * y / x, x̄, x̄, x, y, ȳ))),
    # (maximum,
    #     :(x̄ = broadcast!((x, y, ȳ)->ȳ * (y == x), similar(x), x, y, ȳ)),
    #     :(broadcast!((x̄, x, y, ȳ)->x̄ + ȳ * (y == x), x̄, x̄, x, y, ȳ))),
    # (minimum,
    #     :(x̄ = broadcast!((x, y, ȳ)->ȳ * (y == x), similar(x), x, y, ȳ)),
    #     :(broadcast!((x̄, x, y, ȳ)->x̄ + ȳ * (y == x), x̄, x̄, x, y, ȳ))),
    # (maxabs,
    #     :(x̄ = broadcast!((x, y)->sign(x) * (y == abs(x)), similar(x), x, y)),
    #     :(broadcast!((x̄, x, y)->x̄ + sign(x) * (y == abs(x)), x̄, x̄, x, y))),
    # (minabs,
    #     :(x̄ = broadcast!((x, y)->sign(x) * (y == abs(x)), similar(x), x, y)),
    #     :(broadcast!((x̄, x, y)->x̄ + sign(x) * (y == abs(x)), x̄, x̄, x, y))),
]

# Each of the reduce operations. Both forms are supported.
for (f, new_x̄, update_x̄) in reduce
    fs = Symbol(f)

    # Define the single argument sensitivity.
    @eval @sensitivity $fs{T<:Union{AbstractArray, Real}}(x::T) (x̄, $new_x̄, $update_x̄) y ȳ

    # Define the multiple-argument sensitivity.
    @eval @sensitivity $fs(x::AbstractArray, region) [(x̄, $new_x̄, $update_x̄), ()] y ȳ
end
