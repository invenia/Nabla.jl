import Base.LinAlg.BLAS: gemv, gemv!, gemm!, trsm!, Cholesky, axpy!, ger!
import Base.LinAlg.chol

"""
Returns views to various bits of A. See "Differentiation of the Cholesky decomposition",
Murray 2016 for details.

Arguments:
A - square matrix.
j - row / col index.

Returns:
r - j^th row of A up to (not including) its j^th element.
d - j^th diagonal element of A.
B - block of A contained within r and c.
c - j^th column of A from j+1 to end.
"""
function level2partition(A, j)

    # Check that A is square and j is a valid index.
    M, N = size(A)
    (0 >= j || j > M) && throw(ArgumentError("j is out of range."))
    M != N && throw(ArgumentError("A is not square."))

    # Compute the required views into A.
    r = view(A, j, 1:j-1)
    d = view(A, j, j)
    B = view(A, j+1:N, 1:j-1)
    c = view(A, j+1:N, j)
    return r, d, B, c
end


"""
Returns views into varous bits of A. See "Differentiation of the Cholesky decomposition",
Murray 2016 for details.

Arguments:
A - square matrix. Only the lower triangle will be accessed.
j - first index.
k - second index.

Returns:
LAPACK partitions of A. See "Differentiation of the Cholesky decomposition", Murray 2016 for
details.
"""
function level3partition(A, j, k)

    # Check that A is square and j is a valid index.
    M, N = size(A)
    (0 >= j || j > M) && throw(ArgumentError("j is out of range."))
    M != N && throw(ArgumentError("A is not square."))

    # Get views into bits of A.
    R = view(A, j:k, 1:j-1)
    D = view(A, j:k, j:k)
    B = view(A, k+1:N, 1:j-1)
    C = view(A, k+1:N, j:k)
    return R, D, B, C
end


"""
Compute the unblocked sensitivities of the Cholesky factorisation.

Arguments:
Σ̄ - dΣ. Initial values will be overwritten.
L - Cholesky factorisation. Lower-triangular.

Returns:
Σ̄ - dΣ.
"""
function chol_unblocked_rev!(Σ̄::AbstractMatrix{T}, L::AbstractMatrix{T}) where T<:Real

    # Check that L is square, that Σ̄ is square and that they are the same size.
    M, N = size(Σ̄)
    M != N && throw(ArgumentError("Σ̄ is not square."))

    # Compute the reverse-mode diff.
    j = N
    for ĵ in 1:N
        r, d, B, c = level2partition(L, j)
        r̄, d̄, B̄, c̄ = level2partition(Σ̄, j)

        # d̄ <- d̄ - c'c̄ / d.
        d̄[1] -= dot(c, c̄) / d[1]

        # [d̄ c̄'] <- [d̄ c̄'] / d.
        d̄[1] /= d[1]
        c̄[:] /= d[1]

        # r̄ <- r̄ - [d̄ c̄'] [r' B']'.
        r̄ = gemv!('T', -1.0, B, c̄, 1.0, axpy!(-Σ̄[j, j], r, r̄))

        # B̄ <- B̄ - c̄ r.
        B̄ = BLAS.ger!(-1.0, c̄, copy(r), B̄)
        d̄[1] /= 2
        j -= 1
    end
    return tril!(Σ̄)
end
chol_unblocked_rev(Σ̄::AbstractMatrix, L::AbstractMatrix) = chol_unblocked_rev!(copy(Σ̄), L)

"""
Compute the unblocked sensitivities of the Cholesky factorisation using a blocked,
cache-friendly routine.

Arguments:
Σ̄ - dΣ. Initial values will be overwritten.
L - Cholesky factorisation. Lower-triangular.

Returns:
Σ̄ - dΣ.
"""
function chol_blocked_rev!(Σ̄::AbstractMatrix{T}, L::AbstractMatrix{T}, Nb::Int) where T<:Real

    # Check that L is square, that Σ̄ is square and that they are the same size.
    M, N = size(Σ̄)
    M != N && throw(ArgumentError("Σ̄ is not square."))

    # Compute the reverse-mode diff.
    k = N
    for k̂ in 1:Nb:N

        j = max(1, k - Nb + 1)
        R, D, B, C = level3partition(L, j, k)
        R̄, D̄, B̄, C̄ = level3partition(Σ̄, j, k)

        C̄ = trsm!('R', 'L', 'N', 'N', 1.0, D, C̄)
        gemm!('N', 'N', -1.0, C̄, R, 1.0, B̄)
        gemm!('T', 'N', -1.0, C̄, C, 1.0, D̄)
        chol_unblocked_rev!(D̄, D)
        gemm!('T', 'N', -1.0, C̄, B, 1.0, R̄)
        gemm!('N', 'N', -1.0, D̄ + D̄', R, 1.0, R̄)

        k -= Nb
    end
    return tril!(Σ̄)
end
function chol_blocked_rev(Σ̄::AbstractMatrix, L::AbstractMatrix, Nb::Int)
    return chol_blocked_rev!(copy(Σ̄), L, Nb)
end
