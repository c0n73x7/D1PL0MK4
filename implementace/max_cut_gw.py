import numpy as np
from numpy.linalg import norm, cholesky
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense


def test_graph():
    return np.array([
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0]])


def solve_sdp_program(W):
    assert W.ndim == 2
    assert W.shape[0] == W.shape[1]
    W = W.copy()
    n = W.shape[0]
    with Model('gw_max_cut') as M:
        W = Matrix.dense(W / 4.)
        J = Matrix.ones(n, n)
        # variable
        Y = M.variable('Y', Domain.inPSDCone(n))
        # objective function
        M.objective(ObjectiveSense.Maximize, Expr.dot(W, Expr.sub(J, Y)))
        # constraints
        for i in range(n):
            M.constraint(f'c_{i}', Y.index(i, i), Domain.equalsTo(1.))
        # solve
        M.solve()
        # solution
        Y_opt = Y.level()
    return np.reshape(Y_opt, (n,n))


def find_cut(L, W):
    assert L.ndim == W.ndim == 2
    assert L.shape[0] == L.shape[1] == W.shape[0] == W.shape[1]
    L = L.copy()
    W = W.copy()
    n = L.shape[0]
    # generate random vector on ndim-sphere
    r = np.random.normal(0, 1, n)
    r /= norm(r)
    # find cut
    S, S_comp = list(), list()
    for i in range(n): 
        if np.dot(L[i,:], r) >= 0:
            S.append(i)
        else:
            S_comp.append(i)
    # sum of cut weights
    s = 0
    for i in S:
        for j in S_comp:
            s += W[i][j]
    return s


if __name__ == "__main__":
    W = test_graph()
    relax = solve_sdp_program(W)
    L = cholesky(relax)
    sums = list()
    for _ in range(1000):
        s = find_cut(L, W)
        sums.append(s)
    print(max(sums))
