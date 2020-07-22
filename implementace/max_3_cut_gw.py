import numpy as np
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense


def solve_sdp_program(W, k):
    assert W.ndim == 2
    assert W.shape[0] == W.shape[1]
    W = W.copy()
    n = W.shape[0]
    with Model('gw_max_3_cut') as M:
        W = Matrix.dense(2 / 3 * W)
        J = Matrix.ones(n, n)
        # variable
        Y = M.variable('Y', Domain.inPSDCone(3*n))
        # objective function - TODO: SLICE na Y
        M.objective(ObjectiveSense.Maximize, Expr.dot(W, Expr.sub(J, Y)))
        # constraints
        for i in range(3*n):
            M.constraint(f'c_{i},{i}', Y.index(i, i), Domain.equalsTo(1.))
        for i in range(n):
            M.constraint(f'c_{i}^a^b', Y.index(i*3,   i*3+1), Domain.equalsTo(-1/2.))
            M.constraint(f'c_{i}^a^b', Y.index(i*3,   i*3+2), Domain.equalsTo(-1/2.))
            M.constraint(f'c_{i}^a^b', Y.index(i*3+1, i*3+2), Domain.equalsTo(-1/2.))
        for i in range(n):
            for j in range(n):
                pass
