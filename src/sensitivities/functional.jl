import Base: mapreduce, mapreducedim, map, broadcast, Any16, +, *

"""
    mapreduce{T<:AbstractArray}(f, op, A::Node{T})
Produce optimised implementations of sensitivities for mapreduce for a variety of `op`
parameters. If an implementation of the reverse-mode sensitivities of `f` already exists
then this is used, otherwise sensitivities are automatically generated using the
ForwardDiff.jl package (this may change in the future when a statically-compiled RMAD
package becomes viable to compile these expressions down efficiently).
"""
function mapreduce(f, ::typeof(+), A::Node{T}) where T<:AbstractArray
    if method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Any, Real, Any, Any})
        println("Aha, method exists!")
    else
        println("boooo, no such method exists!")
    end
end

# eval(sensitivity(
#     :(mapreduce(identity, ::typeof(+), A::AbstractArray)),
#     [(), (), (:Ā, :(ones(A) .* Ȳ), :(Ā .= Ā .+ Ȳ))],
#     :Y, :Ȳ))

mapreduce{T<:Number}(f, op, a::Node{T}) = f(a)

# It is assumed that the cardinality of itr is relatively small in the methods below and]
# that there is therefore no need to optimise them.
# mapreduce(f, op, itr), mapreduce(f, op, v0, itr)

# # Reverse-mode sensitivities for each of the mapreducedim methods.
# eval(sensitivity(:(mapreducedim(f, op, A::AbstractArray, region)), ))
# eval(sensitivity(:(mapreducedim(f, op, A::AbstractArray, region, v0)), ))

# # Reverse-mode sensitivities for mapping operations involving tuples.
# eval(sensitivity(:(map(f, t::Tuple{Any})), ))
# eval(sensitivity(:(map(f, t::Tuple{Any, Any})), ))
# eval(sensitivity(:(map(f, t::Tuple{Any, Any, Any})), ))
# eval(sensitivity(:(map(f, t::Tuple)), ))
# eval(sensitivity(:(map(f, t::Any16)), ))

# eval(sensitivity(:(map(f, t::Tuple{Any}, s::Tuple{Any})), ))
# eval(sensitivity(:(map(f, t::Tuple{Any, Any}, s::Tuple{Any, Any})), ))
# eval(sensitivity(:(map(f, t::Tuple, s::Tuple)), ))
# eval(sensitivity(:(map(f, t::Any16, s::Any16)), ))
# eval(sensitivity(:(map(f, t1::Tuple, t2::Tuple, ts::Tuple...)), ))

# eval(sensitivity(:(map(f, t1::Any16, t2::Any16, ts::Any16...)), ))

# # Reverse-mode sensitivities for mapping operations involving numbers and arrays.
# eval(sensitivity(:(map(f, x::Number, ys::Number...)), ))
# eval(sensitivity(:(map(f, rowvecs::RowVector...)), ))
# eval(sensitivity(:(map(f, A::Union{AbstractArray, AbstractSet, Associative})), ))
# eval(sensitivity(:(map(f, A)), ))
# eval(sensitivity(:(map(f, iters...)), ))

# function map(f, A::Node{V}, B::Vararg{Node{T}} where V<:AbstractArray)
#     if method_exists(f, Tuple{eltype(A.val), map(x->eltype(x.val), B...)})
#         println("method exists!")
#     else
#         println("No such method exists... boooooo!")
#     end
# end

# # NOTE: FOR MAPPING OPERATIONS INVOLVING NON-ARRAY OBJECTS, NEED TO THINK ABOUT HOW THE
# # IMPLEMENTATION SHOULD WORK. THE OPTIMISATIONS WILL ONLY REDUCE MEMORY OVERHEAD SLIGHTLY IN
# # THE CASE OF SMALL TUPLES. IF ONE HAS A LARGE ARRAY OF ARRAYS, OPTIMISATIONS MIGHT MAKE
# # SENSE THOUGH. THIS REQUIRES A LOT OF CARE.

# # Sensitivities for broadcasted operations.
# eval(sensitivity(:(broadcast(::Base.*, x::Number, J::UniformScaling)), ))
# eval(sensitivity(:(broadcast(::Base.*, J::UniformScaling, x::Number)), ))

# eval(sensitivity(:(broadcast(::Base./, x::Number, J::UniformScaling)), ))
# eval(sensitivity(:(broadcast(::Base./, J::UniformScaling, x::Number)), ))

# eval(sensitivity(:(broadcast(f, x::Number...)), ))
# eval(sensitivity(:(broadcast(f, t::Tuple{Vararg{Any,N}}, ts::Tuple{Vararg{Any,N}}...)), ))
# eval(sensitivity(:(broadcast(f, rowvecs::Union{Number, RowVector}...)), ))
# eval(sensitivity(:(broadcast(f, A, Bs...)), ))
