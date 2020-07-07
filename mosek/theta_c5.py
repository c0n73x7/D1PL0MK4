import numpy as np
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense
from math import sqrt


MASK = np.zeros((5,5))
MASK[0,1] = MASK[1,0] = 1
MASK[1,2] = MASK[2,1] = 1
MASK[2,3] = MASK[3,2] = 1
MASK[3,4] = MASK[4,3] = 1
MASK[4,0] = MASK[0,4] = 1


with Model('theta_c5') as M:
    MASK = Matrix.dense(MASK)
    # variable
    X = M.variable('X', Domain.inPSDCone(5))
    # objective function
    M.objective(ObjectiveSense.Maximize, Expr.sum(Expr.dot(Matrix.ones(5, 5), X)))
    # constraints
    M.constraint(f'c1', Expr.sum(Expr.dot(X, MASK)), Domain.equalsTo(0.))
    M.constraint(f'c2', Expr.sum(Expr.dot(X, Matrix.eye(5))), Domain.equalsTo(1.))
    # solve
    M.solve()
    # solution
    sol = X.level()

print(sum(sol))
print(sqrt(5))
