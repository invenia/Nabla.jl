using Nabla: checkpoint

function _checkpoint_foo(x)
    y = checkpoint(sin, (x,))
    return cos(y)
end
_foo(x) = cos(sin(x))

_bar(x; y=5.0) = sin(x * y)

@testset "checkpointing" begin

    # Positional args testing.
    @test checkpoint(sin, (5.0,)) == sin(5.0)
    @test ∇(x->checkpoint(sin, (x,)))(5.0) == (cos(5.0),)

    @test _checkpoint_foo(5.0) == _foo(5.0)
    @test ∇(_checkpoint_foo)(5.0) == ∇(_foo)(5.0)

    # kwargs testing.
    @test checkpoint(_bar, (5.0,), (y=4.5,)) == _bar(5.0; y=4.5)
    @test ∇(x->checkpoint(_bar, (x,), (y=4.5,)))(5.0) == ∇(x->_bar(x; y=4.5))(5.0)

    # Check that closures error.
    y = 5.0
    bar(x) = x + y
    @test_throws ErrorException checkpoint(bar, (5.0,))
end
