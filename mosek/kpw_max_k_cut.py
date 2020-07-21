import numpy as np
from numpy.linalg import cholesky, norm
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense, WorkStack
from tqdm import tqdm


def test_graph():
    return np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]])


def generate_random_graph(n, seed=23):
    np.random.seed(seed)
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            W[i,j] = W[j,i] = np.random.randint(2)
    return W


def solve_sdp_program(A):
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    A = A.copy()
    n = A.shape[0]
    with Model('theta_comp') as M:
        #A = Matrix.dense(A.tolist())
        # variables
        U = M.variable('U', Domain.inPSDCone(n))
        t = M.variable('t', Domain.greaterThan(2.))
        # constraints
        #print(Expr.reshape(Expr.repeat(Expr.sub(1.,t), n*n, 0), [n,n]).getShape())
        M.constraint(
            'xxxx',
            Expr.mulElm(
                Expr.mulElm(U, Matrix.dense(A)),
                Expr.eval(Expr.reshape(Expr.repeat(Expr.sub(1.,t), n*n, 0), [n,n]))),
            Domain.equalsTo(Matrix.dense(A)))
        for i in range(n):
            M.constraint(f'c{i},{i}', U.index(i, i), Domain.equalsTo(1.))
        
            #for j in range(i+1, n):
            #    if A[i,j] != 0:
            #        M.constraint(f'c{i},{j}', Expr.mul(U.index(i,j), Expr.sub(1, t),), Domain.equalsTo(1.))
            #        #pow_inv(M, U.index(i,j), Expr.sub(1., t), 1.)
        # objective function
        M.objective(ObjectiveSense.Minimize, t)
        # solve
        M.solve()
        U_sol = U.level()
        t_sol = t.level()
    return t_sol, np.reshape(U_sol, (n,n))


def pow_inv(M, t, x, p):
    M.constraint(Expr.hstack(t, x, 1), Domain.inPPowerCone(1.0/(1.0+p)))


def find_partition(V, A, k):
    assert V.ndim == A.ndim == 2
    assert V.shape[0] == V.shape[1] == A.shape[0] == A.shape[1]
    V = V.copy()
    A = A.copy()
    n = V.shape[0]
    # random vectors
    random_vectors = [np.random.normal(0, 1, n) for _ in range(k)]
    random_vectors = [v / norm(v) for v in random_vectors]
    # find partition
    labels = list()
    for i in range(n):
        prods = np.array([np.dot(V[i,:], r) for r in random_vectors])
        labels.append(np.argmax(prods))
    labels = np.array(labels)
    # sum of weights
    s = 0
    for l in range(k):
        for i in np.argwhere(labels == l).flatten():
            for j in np.argwhere(labels != l).flatten():
                s += A[i][j]
    return {
        'labels': labels,
        'sum': s/2.,
    }


if __name__ == "__main__":
    # A = generate_random_graph(100)
    A = test_graph()
    k = 3
    n = A.shape[0]
    t_relax, U_relax = solve_sdp_program(A)
    print(t_relax)
    #t = max(k, ceil(t_relax))
    #V = cholesky(U_relax)
    #sums = list()
    #best_sum = -1
    #for _ in tqdm(range(1000)):
    #    res = find_partition(V, A, k)
    #    s = res.get('sum')
    #    sums.append(s)
    #    if s > best_sum:
    #        best_sum = s
    #print(f'best sum of cut weights is {best_sum}')
    # print(sums)
