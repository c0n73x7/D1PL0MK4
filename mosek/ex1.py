import json
import numpy as np
from mosek.fusion import Model, Domain, Expr, ObjectiveSense


A = np.array([[-1., 3.], [4., -1.]])
b = np.array([4., 6.])
c = np.array([1., 1.])


with Model('ex1') as M:
    # variable x
    x = M.variable('x', 2, Domain.greaterThan(0.))
    # constraints
    M.constraint('c1', Expr.dot(A[0, :], x), Domain.lessThan(b[0]))
    M.constraint('c2', Expr.dot(A[1, :], x), Domain.lessThan(b[1]))
    # objective function
    M.objective('obj', ObjectiveSense.Maximize, Expr.dot(c, x))
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
