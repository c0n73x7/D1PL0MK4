import numpy as np
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense


def test_graph():
    return np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]])


def solve_sdp_program(A):
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    A = A.copy()
    n = A.shape[0]
    with Model('theta_2') as M:
        # variable
        X = M.variable('X', Domain.inPSDCone(n+1))
        t = M.variable()
        # objective function
        M.objective(ObjectiveSense.Maximize, t)
        # constraints
        for i in range(n+1):
            M.constraint(f'c{i}{i}', X.index(i, i), Domain.equalsTo(1.))
            if i == 0:
                continue
            M.constraint(f'c0,{i}', Expr.sub(X.index(0, i), t), Domain.greaterThan(0.))
            for j in range(i+1, n+1):
                if A[i-1,j-1] == 0:
                    M.constraint(f'c{i},{j}', X.index(i,j), Domain.equalsTo(0.))
        # solution
        M.solve()
        X_sol = X.level()
        t_sol = t.level()
    t_sol = t_sol[0]
    theta = 1. / t_sol**2
    return theta


if __name__ == "__main__":
    A = test_graph()
    print(solve_sdp_program(A))
