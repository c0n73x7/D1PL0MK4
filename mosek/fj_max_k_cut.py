import numpy as np
from numpy.linalg import cholesky
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense
from tqdm import tqdm


def generate_random_graph(n, seed=23):
    np.random.seed(seed)
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            W[i,j] = W[j,i] = np.random.randint(2)
    return W


def solve_sdp_program(W, k):
    assert W.ndim == 2
    assert W.shape[0] == W.shape[1]
    W = W.copy()
    n = W.shape[0]
    with Model('fj_max_k_cut') as M:
        W = Matrix.dense((k - 1) / k * W)
        J = Matrix.ones(n, n)
        # variable
        Y = M.variable('Y', Domain.inPSDCone(n))
        # objective function
        M.objective(ObjectiveSense.Maximize, Expr.dot(W, Expr.sub(J, Y)))
        # constraints
        for i in range(n):
            for j in range(i + 1, n):
                M.constraint(f'c_{i},{j}', Y.index(i, j), Domain.greaterThan(-1 / (k - 1)))
        for i in range(n):
            M.constraint(f'c_{i}', Y.index(i, i), Domain.equalsTo(1.))
        # solve
        M.solve()
        # solution
        Y_opt = Y.level()
    return np.reshape(Y_opt, (n,n))


def find_partition(A, W, k):
    assert A.ndim == W.ndim == 2
    assert A.shape[0] == A.shape[1] == W.shape[0] == W.shape[1]
    A = A.copy()
    W = W.copy()
    n = A.shape[0]
    # random vectors
    random_vectors = [np.random.normal(0, 1, n) for _ in range(k)]
    # find partition
    labels = list()
    for i in range(n):
        prods = np.array([np.dot(A[i,:], g) for g in random_vectors])
        labels.append(np.argmax(prods))
    labels = np.array(labels)
    # sum of weights
    s = 0
    for l in range(k):
        for i in np.argwhere(labels == l).flatten():
            for j in np.argwhere(labels != l).flatten():
                s += W[i][j]
    return {
        'labels': labels,
        'sum': s/2.,
    }


if __name__ == "__main__":
    W = generate_random_graph(100)
    k = 7
    RELAX = solve_sdp_program(W, k)
    A = cholesky(RELAX)
    sums = list()
    best_sum = -1
    for _ in tqdm(range(10)):
        res = find_partition(A, W, k)
        s = res.get('sum')
        sums.append(s)
        if s > best_sum:
            best_sum = s
    print(f'best sum of cut weights is {best_sum}')
    # print(sums)
