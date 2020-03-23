import json
import numpy as np
from mosek.fusion import Model, Domain, Expr, ObjectiveSense


A = np.array([[-1., 3., 1., 0.], [4., -1., 0., 1.]])
b = np.array([4., 6.])
c = np.array([1., 1., 0., 0.])


with Model('ex2') as M:
    # variable y
    y = M.variable('y', 2, Domain.greaterThan(0.))
    # constraints
    M.constraint('c1', Expr.dot(A.T[0, :2], y), Domain.greaterThan(c[0]))
    M.constraint('c2', Expr.dot(A.T[1, :2], y), Domain.greaterThan(c[1]))
    # objective function
    M.objective('obj', ObjectiveSense.Minimize, Expr.dot(b, y))
    # solve
    M.solve()
    # solution
    sol = y.level()

# report to json
with open('ex2_output.json', 'w') as f:
    json.dump({
        'solution': {f'y[{i+1}]': yi for i, yi in enumerate(sol)},
        'cost': np.dot(b, sol),
    }, f)
