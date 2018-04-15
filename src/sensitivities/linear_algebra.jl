# Import all linear algebra optimisations from DiffLinearAlgebra.
for op in DLA.ops
    @eval $(import_expr(op))
    @eval @primitive $(op.f)(x...) where __CONTEXT__ <: ∇Ctx =
        propagate_forward($(op.f), x...)
end
