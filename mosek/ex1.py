import numpy as np
import json
from mosek.fusion import *


A = np.array([[-1., 3., 1., 0.], [4., -1., 0., 1.]])
b = np.array([4., 6.])
c = np.array([1., -8, 0., 0.])

with Model('ex1') as M:
    # variable x
    x = M.variable('x', 4, Domain.greaterThan(0.))
    # constraints
    M.constraint('c1', Expr.dot(A[0, :], x), Domain.equalsTo(b[0]))
    M.constraint('c2', Expr.dot(A[1, :], x), Domain.equalsTo(b[1]))
    # objective function
    M.objective('obj', ObjectiveSense.Minimize, Expr.dot(c, x))
    # solve
    M.solve()
    # solution
    sol = x.level()

# report to json
with open('ex1_output.json', 'w') as f:
    json.dump({
        'solution': {f'x[{i+1}]': xi for i, xi in enumerate(sol)},
        'cost': np.dot(sol, c),
    }, f)
