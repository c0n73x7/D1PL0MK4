import numpy as np
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense
from itertools import product
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


def expand_matrix(A):
    A = A.copy()
    n = A.shape[0]
    B = np.zeros((3*n, 3*n))
    for i in range(n):
        for j in range(n):
            B[i*3,j*3] = A[i,j]
    return B


def solve_sdp_program(W):
    assert W.ndim == 2
    assert W.shape[0] == W.shape[1]
    W = W.copy()
    n = W.shape[0]
    W = expand_matrix(W)
    with Model('gw_max_3_cut') as M:
        W = Matrix.dense(W / 3.)
        J = Matrix.ones(3*n, 3*n)
        # variable
        Y = M.variable('Y', Domain.inPSDCone(3*n))
        # objective function
        M.objective(ObjectiveSense.Maximize, Expr.dot(W, Expr.sub(J, Y)))
        # constraints
        for i in range(3*n):
            M.constraint(f'c_{i}{i}', Y.index(i, i), Domain.equalsTo(1.))
        for i in range(n):
            M.constraint(f'c_{i}^01', Y.index(i*3,   i*3+1), Domain.equalsTo(-1/2.))
            M.constraint(f'c_{i}^02', Y.index(i*3,   i*3+2), Domain.equalsTo(-1/2.))
            M.constraint(f'c_{i}^12', Y.index(i*3+1, i*3+2), Domain.equalsTo(-1/2.))
            for j in range(i+1, n):
                for a, b in product(range(3), repeat=2):
                    M.constraint(f'c_{i}{j}^{a}{b}', Y.index(i*3 + a, j*3 + b), Domain.greaterThan(-1/2.))
                    M.constraint(f'c_{i}{j}^{a}{b}1', Expr.sub(Y.index(i*3 + a, j*3 + b), Y.index(i*3 + (a + 1) % 3, j*3 + (b + 1) % 3)), Domain.equalsTo(0.))
                    M.constraint(f'c_{i}{j}^{a}{b}2', Expr.sub(Y.index(i*3 + a, j*3 + b), Y.index(i*3 + (a + 2) % 3, j*3 + (b + 2) % 3)), Domain.equalsTo(0.))
        # solution
        M.solve()
        Y_opt = Y.level()
    return np.reshape(Y_opt, (3*n,3*n))


def find_partition(L, W):
    pass


if __name__ == "__main__":
    W = test_graph()
    relax = solve_sdp_program(W)
    L = cholesky(relax)
    print(L)
