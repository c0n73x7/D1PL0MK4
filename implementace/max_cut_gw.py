import numpy as np
from numpy.linalg import norm
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense


def generate_random_graph(n, seed=23):
    np.random.seed(seed)
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            W[i,j] = W[j,i] = np.random.randint(2)
    return W


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


def find_cut(A, W):
    assert A.ndim == W.ndim == 2
    assert A.shape[0] == A.shape[1] == W.shape[0] == W.shape[1]
    A = A.copy()
    W = W.copy()
    n = A.shape[0]
    # generate random vector on ndim-sphere
    r = np.random.normal(0, 1, n)
    r /= norm(r)
    # find cut
    S, S_comp = list(), list()
    for i in range(n): 
        if np.dot(A[i,:], r) >= 0:
            S.append(i)
        else:
            S_comp.append(i)
    # sum of cut weights
    s = 0
    for i in S:
        for j in S_comp:
            s += W[i][j]
    return s
