# Import all linear algebra optimisations from DiffLinearAlgebra.
for op in DLA.ops
    @eval $(import_expr(op))
    @eval @explicit_intercepts $(op.f) $(op.T)
end
