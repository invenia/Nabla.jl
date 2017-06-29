@testset "sensitivity_sig" begin

    # Ensure that the expression produced by to_expr evals back to an equivalent UnionAll.
    function check_to_expr(u::Type)
        expr = AutoGrad2.to_expr(u)
        return eval(expr) == u
    end
    @test check_to_expr(Tuple{Real})
    @test check_to_expr(Tuple{T} where T)
    @test check_to_expr(Tuple{T} where T<:Real)
    @test check_to_expr(Tuple{T, T} where T)
    @test check_to_expr(Tuple{T, T} where T<:Number)
    @test check_to_expr(Tuple{T, Real} where T)
    @test check_to_expr(Tuple{Number, Number})
    @test check_to_expr(Tuple{T, T, V, Real} where T where V<:Number)
    @test check_to_expr(Tuple{T, T, V, Real} where {T, V<:Number})
    @test check_to_expr(Tuple{Union{T, Nullable{T}}} where T)
    @test check_to_expr(Tuple{Union{T, Nullable{T}}} where T<:(Nullable{A} where A))
    @test check_to_expr(Tuple{Union{T, Nullable{T}}} where T<:(Nullable{A} where A<:Real))

    # Ensure that the correct Symbols are returned.
    function check_typesyms_from_sig(t, expected)
        return AutoGrad2.typesyms_from_sig(t) == expected
    end
    @test check_typesyms_from_sig(Tuple{Real}, [:Real])
    @test check_typesyms_from_sig(Tuple{T} where T<:Number, [:T])
    @test check_typesyms_from_sig(Tuple{T, Real} where T, [:T, :Real])
    @test check_typesyms_from_sig(Tuple{Real, T} where T, [:Real, :T])
    @test check_typesyms_from_sig(Tuple{T, V, Q} where {T, V, Q}, [:T, :V, :Q])
    # @test check_typesyms_from_sig(
    #     Tuple{Nullable{T}} where T<:Real,
    #     [:(Nullable{T})]
    # )

end
