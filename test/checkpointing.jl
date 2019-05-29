using Nabla: checkpoint

function _checkpoint_foo(x)
    y = checkpoint(sin, (x,))
    return cos(y)
end
_foo(x) = cos(sin(x))

@testset "checkpointing" begin
    @test ∇(x->checkpoint(sin, (x,)))(5.0) == (cos(5.0),)
    @test ∇(_checkpoint_foo)(5.0) == ∇(_foo)(5.0)
end
