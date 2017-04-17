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


# Sensitivities for elementwise versions of binary operators of the form z = x (op) y.
binary_sensitivities_elementwise = [
    (:.+,
        :(x̄ = broadcastsum!(z̄->z̄, false, similar(x), z̄)),
        :(ȳ = broadcastsum!(z̄->z̄, false, similar(y), z̄)),
        :(x̄ = broadcastsum!(z̄->z̄, true, x̄, z̄)),
        :(ȳ = broadcastsum!(z̄->z̄, true, ȳ, z̄)),
        (lb, ub), (lb, ub)),
    (:.-,
        :(x̄ = broadcastsum!(z̄->z̄, false, similar(x), z̄)),
        :(ȳ = broadcastsum!(z̄->-z̄, false, similar(y), z̄)),
        :(x̄ = broadcastsum!(z̄->z̄, true, x̄, z̄)),
        :(ȳ = broadcastsum!(z̄->-z̄, true, ȳ, z̄)),
        (lb, ub), (lb, ub)),
    (:.*,
        :(x̄ = broadcastsum!((y, z̄)->y * z̄, false, similar(x), y, z̄)),
        :(ȳ = broadcastsum!((x, z̄)->x * z̄, false, similar(y), x, z̄)),
        :(x̄ = broadcastsum!((y, z̄)->y * z̄, true, x̄, y, z̄)),
        :(ȳ = broadcastsum!((x, z̄)->x * z̄, true, ȳ, x, z̄)),
        (lb, ub), (lb, ub)),
    (:./,
        :(x̄ = broadcastsum!((y, z̄)->z̄ / y, false, similar(x), y, z̄)),
        :(ȳ = broadcastsum!((x, y, z̄)->-z̄ * x / y^2, false, similar(y), x, y, z̄)),
        :(x̄ = broadcastsum!((y, z̄)->z̄ / y, true, x̄, y, z̄)),
        :(ȳ = broadcastsum!((x, y, z̄)->-z̄ * x / y^2, true, ȳ, x, y, z̄)),
        (lb, ub), (lb, ub)),
    (:.\,
        :(x̄ = broadcastsum!((x, y, z̄)-> -z̄ * y / x^2, false, similar(x), x, y, z̄)),
        :(ȳ = broadcastsum!((x, z̄)->z̄ / x, false, similar(y), x, z̄)),
        :(x̄ = broadcastsum!((x, y, z̄)-> -z̄ * y / x^2, true, x̄, x, y, z̄)),
        :(ȳ = broadcastsum!((x, z̄)->z̄ / x, true, ȳ, x, z̄)),
        (lb, ub), (lb, ub)),
]

for (f, new_x̄, new_ȳ, update_x̄, update_ȳ, xr, yr) in binary_sensitivities_elementwise
    eval(compute_sensitivity_method(f,
        [:(T <: ArrayOrFloat), :(U <: ArrayOrFloat)],
        [:x, :y], [:x̄, :ȳ], [:T, :U], [true, true],
        :z, :z̄, [new_x̄, new_ȳ], [update_x̄, update_ȳ]
    ))
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
