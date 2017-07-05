# Implementation of functionals (i.e. higher-order functions).

# Intercepts for mapreduce.
add_intercept(:mapreduce, :Base)
∇_functionals[:mapreduce] = quote
    function AutoGrad2.∇(
        ::typeof(mapreduce),
        ::Type{Arg{3}},
        p, y, ȳ, f,
        ::typeof($(esc(:+))),
        A::AbstractArray{T} where T<:Real,
    )
        println(f)
        if needs_output(f) && method_exists(∇, Tuple{typeof(f), Arg{1}, Real, Any})
            return map(a->ȳ * ∇(f, Arg{1}, a, y), A)
        elseif !needs_output(f) && method_exists(∇, Tuple{typeof(f)}, Arg{1}, Real)
            return map(a->ȳ * ∇(f, Arg{1}, a), A)
        else
            throw(error("Not implemented mapreduce sensitivities for general f."))
        end
    end
end

# function ∇(::typeof(mapreduce), ::Type{Arg{3}}, p, y, ȳ, f, ::typeof(*), A::Real)

# end

# intercepts[mapreduce] = quote

# @generated function mapreduce(f, ::typeof(+), A::Node{T}) where T<:AbstractArray
#     if method_exists(∇, Tuple{f, Type{Arg{1}}, Any, Real, Any, Any})
#         if needs_output(f)
#             return quote
#                 map(f, )
#             end
#         else

#         end
#     else
#         throw(error("Not implemented automatic creation of sensitivities for +."))
#     end
# end



# function mapreduce(f, ::typeof(*), A::Node{T} where T<:AbstractArray)
#     if method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Any, Real, Any, Any})
#         println("Aha, method exists!")
#     else
#         println("boooo, no such method exists!")
#     end
# end

# end




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
