import numpy as np
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense


MASK = np.zeros((7,7))
MASK[0,1] = MASK[1,0] = 1
MASK[1,2] = MASK[2,1] = 1
MASK[2,3] = MASK[3,2] = 1
MASK[3,4] = MASK[4,3] = 1
MASK[4,5] = MASK[5,4] = 1
MASK[5,6] = MASK[6,5] = 1
MASK[6,0] = MASK[0,6] = 1


with Model('theta_c7') as M:
    MASK = Matrix.dense(MASK)
    # variable
    X = M.variable('X', Domain.inPSDCone(7))
    # objective function
    M.objective(ObjectiveSense.Maximize, Expr.sum(Expr.dot(Matrix.ones(7, 7), X)))
    # constraints
    M.constraint(f'c1', Expr.sum(Expr.dot(X, MASK)), Domain.equalsTo(0.))
    M.constraint(f'c2', Expr.sum(Expr.dot(X, Matrix.eye(7))), Domain.equalsTo(1.))
    # solve
    M.solve()
    # solution
    sol = X.level()

print(sum(sol))
