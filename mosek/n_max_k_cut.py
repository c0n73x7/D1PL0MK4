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
    g = np.random.normal(0, 1, 2*n)
    psi = np.random.uniform(0,2*np.pi)
    # labels
    labels = list()
    for i in range(n):
        vi = A[i,:]
        ui1 = np.concatenate((vi, np.zeros(n, dtype=float)))
        ui2 = np.concatenate((np.zeros(n, dtype=float), vi))
        prod1 = np.dot(g, ui1)
        prod2 = np.dot(g, ui2)
        theta = psi
        if prod1 >= 0 and prod2 >= 0:
            theta += np.arctan(prod2 / prod1)
        elif prod1 <= 0 and prod2 >= 0:
            theta += np.arctan(prod2 / prod1) + np.pi
        elif prod1 <= 0 and prod2 <= 0:
            theta += np.arctan(prod2 / prod1) + np.pi
        elif prod1 >= 0 and prod2 <= 0:
            theta += np.arctan(prod2 / prod1) + 2 * np.pi
        theta = theta % 2*np.pi
        labels.append(int((theta * k) / (2 * np.pi)))
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
    for _ in range(10):
        res = find_partition(A, W, k)
        s = res.get('sum')
        sums.append(s)
        if s > best_sum:
            best_sum = s
    print(f'best sum of cut weights is {best_sum}')
