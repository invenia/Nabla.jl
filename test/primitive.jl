print("primitive.jl... ")

function check_constructboxedfunc()
    fname, args, box = :(sum{T<:AbstractFloat}), [(:x, :(Vector{T}))], [true]
    expr_hand = Expr(:(=), :(sum{T<:AbstractFloat}(x::Node{Vector{T}})), :(Branch(sum, (x,))))
    expr_auto = AutoGrad2.constructboxedfunc(fname, args, box)
    dump(expr_hand)
    dump(expr_auto)
    return isequal(expr_hand, expr_auto)
end
# @test check_constructboxedfunc()

function check_constructboxedfunc_simple()
    fname, arg, box = :(sum), [(:x, :Any)], [true]
    expr_hand = Expr(:(=), :(sum(x::Node{Any})), :(Branch(sum, (x,))))
    expr_auto = AutoGrad2.constructboxedfunc(fname, arg, box)
    return isequal(expr_hand, expr_auto)
end
# @test check_constructboxedfunc_simple()

function check_constructboxedfunc_simplemultipleargs()
    fname, arg, box = :(sum{T}), [(:x, :Any), (:y, :(Vector{T}))], [true, false]
    expr_hand = Expr(:(=), :(sum{T}(x::Node{Any}, y::Vector{T})), :(Branch(sum, (x,y))))
    expr_auto = AutoGrad2.constructboxedfunc(fname, arg, box)
    return isequal(expr_hand, expr_auto)
end
# @test check_constructboxedfunc_simplemultipleargs()

function check_constructboxedfunc_simplemultipleargstrue()
    fname, arg, box = :(sum{T<:AbstractFloat}), [(:x, :Any), (:y, :(Vector{T}))], [true, true]
    expr_hand = Expr(:(=), :(sum{T<:AbstractFloat}(x::Node{Any}, y::Node{Vector{T}})), :(Branch(sum, (x,y))))
    expr_auto = AutoGrad2.constructboxedfunc(fname, arg, box)
    return isequal(expr_hand, expr_auto)
end
# @test check_constructboxedfunc_simplemultipleargstrue()

println("passing.")
