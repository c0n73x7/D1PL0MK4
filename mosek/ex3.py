import json
import numpy as np
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense


def get_sol_report(sol):
    assert sol.ndim == 2
    sol_report = dict()
    for i in range(sol.shape[0]):
        for j in range(sol.shape[1]):
            sol_report[f'x{i+1}{j+1}'] = sol[i, j]
    return sol_report


C = np.array([[2., 1.], [1., 0.]])
A1 = np.eye(2)
b1 = 1.


with Model('ex3') as M:
    C, A1 = Matrix.dense(C), Matrix.sparse(A1)
    # variable X
    X = M.variable('X', Domain.inPSDCone(2))
    # objective function
    M.objective(ObjectiveSense.Minimize, Expr.dot(C, X))
    # constraints
    M.constraint('c1', Expr.dot(A1, X), Domain.equalsTo(b1))
    # solve
    M.solve()
    # solution
    sol = X.level()

# report to json
with open('ex3_output.json', 'w') as f:
    json.dump({
        'solution': get_sol_report(sol.reshape((2,2))),
        'cost': np.dot(C.getDataAsArray(), sol),
    }, f)
