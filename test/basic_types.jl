print("basic_types.jl... ")

function check_scalar()
    return getzero(5.0) == 0.0 && getone(4.933) == 1.0
end
@test check_scalar()

function check_vector()
    return getzero(randn(5)) == zeros(5) && getone(randn(5)) == ones(5)
end
@test check_vector()

function check_tuple()
    t = (5.0, (5.0, randn(2)))
    return getzero(t) == (0.0, (0.0, zeros(2))) && getone(t) == (1.0, (1.0, ones(2)))
end
@test check_tuple()

println("passing.")
