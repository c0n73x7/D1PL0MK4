import sys
import os
import json
import numpy as np
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense


def load_graph(p):
    assert os.path.exists(p), f'file "{p}" not found'
    assert os.path.splitext(p)[1] == '.json', f'"{p}" is not json'
    with open(p, 'r') as f:
        graph = json.load(f)
    n = graph.get('n')
    edges = graph.get('edges')
    A = np.zeros((n, n))
    for e in edges:
        s, e = e[0], e[1]
        A[s,e] = A[e,s] = 1
    return A


if __name__ == "__main__":
    assert len(sys.argv) > 1
    A = load_graph(sys.argv[1])
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    with Model('theta') as M:
        A = Matrix.dense(A)
        # variable
        X = M.variable('X', Domain.inPSDCone(n))
        # objective function
        M.objective(ObjectiveSense.Maximize, Expr.sum(Expr.dot(Matrix.ones(n, n), X)))
        # constraints
        M.constraint(f'c1', Expr.sum(Expr.dot(X, A)), Domain.equalsTo(0.))
        M.constraint(f'c2', Expr.sum(Expr.dot(X, Matrix.eye(n))), Domain.equalsTo(1.))
        # solve
        M.solve()
        # solution
        sol = X.level()
    print(sum(sol))
