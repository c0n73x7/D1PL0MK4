import sys
import os
import json
import numpy as np
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense


def test_graph():
    return np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]])


def load_graph(p):
    assert os.path.exists(p), f'file "{p}" not found'
    assert os.path.splitext(p)[1] == '.json', f'"{p}" is not json'
    with open(p, 'r') as f:
        graph = json.load(f)
    n = graph.get('n')
    A = np.zeros((n, n))
    for e in graph.get('edges'):
        s, e = e[0], e[1]
        A[s,e] = A[e,s] = 1
    return A


def generate_random_graph(n, seed=23):
    np.random.seed(seed)
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            W[i,j] = W[j,i] = np.random.randint(2)
    return W


if __name__ == "__main__":
    #assert len(sys.argv) > 1
    #A = load_graph(sys.argv[1])
    #A = generate_random_graph(100)
    A = test_graph()
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    #A = (A + 1) % 2 - np.eye(n)
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
