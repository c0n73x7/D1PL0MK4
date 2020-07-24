import numpy as np
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense
from numpy.linalg import cholesky


def test_graph():
    return np.array([
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0]])


def solve_sdp_program(W, k):
    assert W.ndim == 2
    assert W.shape[0] == W.shape[1]
    W = W.copy()
    n = W.shape[0]
    with Model('fj_max_k_cut') as M:
        W = Matrix.dense((k - 1) / (2*k) * W)
        J = Matrix.ones(n, n)
        # variable
        Y = M.variable('Y', Domain.inPSDCone(n))
        # objective function
        M.objective(ObjectiveSense.Maximize, Expr.dot(W, Expr.sub(J, Y)))
        # constraints
        for i in range(n):
            M.constraint(f'c_{i}', Y.index(i, i), Domain.equalsTo(1.))
            for j in range(i + 1, n):
                M.constraint(f'c_{i},{j}', Y.index(i, j), Domain.greaterThan(-1 / (k - 1)))
        # solution
        M.solve()
        Y_opt = Y.level()
    return np.reshape(Y_opt, (n,n))


def find_partition(L, W, k, iters=1000):
    assert L.ndim == W.ndim == 2
    assert L.shape[0] == L.shape[1] == W.shape[0] == W.shape[1]
    L = L.copy()
    W = W.copy()
    n = L.shape[0]
    sums = list()
    for _ in range(iters):
        # random vectors
        random_vectors = [np.random.normal(0, 1, n) for _ in range(k)]
        # labels
        labels = list()
        for i in range(n):
            prods = np.array([np.dot(L[i,:], g) for g in random_vectors])
            labels.append(np.argmax(prods))
        labels = np.array(labels)
        # sum
        s = 0
        for l in range(k):
            for i in np.argwhere(labels == l).flatten():
                for j in np.argwhere(labels != l).flatten():
                    s += W[i][j]
        sums.append(int(s/2))
    return max(sums)


if __name__ == "__main__":
    W = test_graph()
    k = 3
    relax = solve_sdp_program(W, k)
    L = cholesky(relax)
    best_sum = find_partition(L, W, k)
    print(best_sum)
