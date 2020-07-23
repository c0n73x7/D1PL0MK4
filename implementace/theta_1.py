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
    with Model('theta_1') as M:
        A = Matrix.dense(A)
        # variable
        X = M.variable('X', Domain.inPSDCone(n))
        # objective function
        M.objective(ObjectiveSense.Maximize, Expr.sum(Expr.dot(Matrix.ones(n, n), X)))
        # constraints
        M.constraint(f'c1', Expr.sum(Expr.dot(X, A)), Domain.equalsTo(0.))
        M.constraint(f'c2', Expr.sum(Expr.dot(X, Matrix.eye(n))), Domain.equalsTo(1.))
        # solution
        M.solve()
        sol = X.level()
    return sum(sol)


if __name__ == "__main__":
    A = test_graph()
    print(solve_sdp_program(A))
