import numpy as np
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense


def solve_sdp_program(W, k):
    assert W.ndim == 2
    assert W.shape[0] == W.shape[1]
    W = W.copy()
    n = W.shape[0]
    with Model('fj_max_k_cut') as M:
        W = Matrix.dense((k - 1) / k * W)
        J = Matrix.ones(n, n)
        # variable
        Y = M.variable('Y', Domain.inPSDCone(n))
        # objective function
        M.objective(ObjectiveSense.Maximize, Expr.dot(W, Expr.sub(J, Y)))
        # constraints
        for i in range(n):
            for j in range(i + 1, n):
                M.constraint(f'c_{i},{j}', Y.index(i, j), Domain.greaterThan(-1 / (k - 1)))
        for i in range(n):
            M.constraint(f'c_{i}', Y.index(i, i), Domain.equalsTo(1.))
        # solve
        M.solve()
        # solution
        Y_opt = Y.level()
    return np.reshape(Y_opt, (n,n))


def find_partition(A, W, k):
    assert A.ndim == W.ndim == 2
    assert A.shape[0] == A.shape[1] == W.shape[0] == W.shape[1]
    A = A.copy()
    W = W.copy()
    n = A.shape[0]
    # random vectors
    random_vectors = [np.random.normal(0, 1, n) for _ in range(k)]
    # labels
    labels = list()
    for i in range(n):
        prods = np.array([np.dot(A[i,:], g) for g in random_vectors])
        labels.append(np.argmax(prods))
    labels = np.array(labels)
    # sum of weights
    s = 0
    for l in range(k):
        for i in np.argwhere(labels == l).flatten():
            for j in np.argwhere(labels != l).flatten():
                s += W[i][j]
    return s/2.
