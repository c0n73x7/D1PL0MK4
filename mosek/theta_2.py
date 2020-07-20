import sys
import os
import json
import numpy as np
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense
from math import sqrt


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
    # A = generate_random_graph(100)
    A = test_graph()
    assert A.shape[0] == A.shape[1]
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
        # solve
        M.solve()
        # solution
        X_sol = X.level()
        t_sol = t.level()
    t_sol = t_sol[0]
    theta = 1. / t_sol**2
    print(theta)
