import numpy as np
from numpy.linalg import cholesky, norm
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense
from tqdm import tqdm


def generate_random_graph(n, seed=23):
    np.random.seed(seed)
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            W[i,j] = W[j,i] = np.random.randint(2)
    return W


def solve_sdp_program(W):
    assert W.ndim == 2
    assert W.shape[0] == W.shape[1]
    W = W.copy()
    n = W.shape[0]
    with Model('gw_max_cut') as M:
        W = Matrix.dense(W / 4.)
        J = Matrix.ones(n, n)
        # variable
        Y = M.variable('Y', Domain.inPSDCone(n))
        # objective function
        M.objective(ObjectiveSense.Maximize, Expr.dot(W, Expr.sub(J, Y)))
        # constraints
        for i in range(n):
            M.constraint(f'c_{i}', Y.index(i, i), Domain.equalsTo(1.))
        # solve
        M.solve()
        # solution
        Y_opt = Y.level()
    return np.reshape(Y_opt, (n,n))


def find_cut(A, W):
    assert A.ndim == W.ndim == 2
    assert A.shape[0] == A.shape[1] == W.shape[0] == W.shape[1]
    A = A.copy()
    W = W.copy()
    n = A.shape[0]
    # generate random vector on ndim-sphere
    r = np.random.normal(0, 1, n)
    r /= norm(r)
    # find cut
    S, S_comp = list(), list()
    for i in range(n): 
        if np.dot(A[i,:], r) >= 0:
            S.append(i)
        else:
            S_comp.append(i)
    # sum of cut weights
    sum_cut_weights = 0
    for i in S:
        for j in S_comp:
            sum_cut_weights += W[i][j]
    return {
        'S': S,
        'S_comp': S_comp,
        'sum_cut_weights': sum_cut_weights,
    }


if __name__ == "__main__":
    W = generate_random_graph(100)
    RELAX = solve_sdp_program(W)
    A = cholesky(RELAX)
    np.random.seed(1)
    eps   = 0.00001
    alpha = 0.87856
    c = (eps * alpha) / (2 + 2 * eps - alpha)
    print(f'for eps={eps}, run step 2 and 3 {int(1/c + 1)} times')
    cut_sums = list()
    best_sum = -1
    best_S, best_S_comp = list(), list()
    for _ in tqdm(range(int(1/c + 1))):
        res = find_cut(A, W)
        cut_sum = res.get('sum_cut_weights')
        cut_sums.append(cut_sum)
        if cut_sum > best_sum:
            best_sum = cut_sum
            best_S = res.get('S')
            best_S_comp = res.get('S_comp')
    print(f'best sum of cut weights is {best_sum}')
