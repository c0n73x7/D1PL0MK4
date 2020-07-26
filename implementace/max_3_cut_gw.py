import numpy as np
from mosek.fusion import Matrix, Model, Domain, Expr, ObjectiveSense
from itertools import product
from numpy.linalg import cholesky, norm
from utils import generate_random_graph


def test_graph():
    return np.array([
        [0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0]])


def expand_matrix(A):
    A = A.copy()
    n = A.shape[0]
    B = np.zeros((3*n, 3*n))
    for i in range(n):
        for j in range(n):
            B[i*3,j*3] = A[i,j]
    return B


def solve_sdp_program(W):
    assert W.ndim == 2
    assert W.shape[0] == W.shape[1]
    W = W.copy()
    n = W.shape[0]
    W = expand_matrix(W)
    with Model('gw_max_3_cut') as M:
        W = Matrix.dense(W / 3.)
        J = Matrix.ones(3*n, 3*n)
        # variable
        Y = M.variable('Y', Domain.inPSDCone(3*n))
        # objective function
        M.objective(ObjectiveSense.Maximize, Expr.dot(W, Expr.sub(J, Y)))
        # constraints
        for i in range(3*n):
            M.constraint(f'c_{i}{i}', Y.index(i, i), Domain.equalsTo(1.))
        for i in range(n):
            M.constraint(f'c_{i}^01', Y.index(i*3,   i*3+1), Domain.equalsTo(-1/2.))
            M.constraint(f'c_{i}^02', Y.index(i*3,   i*3+2), Domain.equalsTo(-1/2.))
            M.constraint(f'c_{i}^12', Y.index(i*3+1, i*3+2), Domain.equalsTo(-1/2.))
            for j in range(i+1, n):
                for a, b in product(range(3), repeat=2):
                    M.constraint(f'c_{i}{j}^{a}{b}-0', Y.index(i*3 + a, j*3 + b), Domain.greaterThan(-1/2.))
                    M.constraint(f'c_{i}{j}^{a}{b}-1', Expr.sub(Y.index(i*3 + a, j*3 + b), Y.index(i*3 + (a + 1) % 3, j*3 + (b + 1) % 3)), Domain.equalsTo(0.))
                    M.constraint(f'c_{i}{j}^{a}{b}-2', Expr.sub(Y.index(i*3 + a, j*3 + b), Y.index(i*3 + (a + 2) % 3, j*3 + (b + 2) % 3)), Domain.equalsTo(0.))
        # solution
        M.solve()
        Y_opt = Y.level()
    return np.reshape(Y_opt, (3*n,3*n))


def find_partition(L, W, iters=1000):
    assert L.ndim == W.ndim == 2
    assert L.shape[0] == L.shape[1]
    assert W.shape[0] == W.shape[1]
    assert L.shape[0] == 3*W.shape[0]
    L = L.copy()
    W = W.copy()
    n = W.shape[0]
    sums = list()
    for _ in range(iters):
        # random vector and random angle
        g = np.random.normal(0, 1, 3*n)
        psi = np.random.uniform(0, 2*np.pi)
        # angles
        angles = list()
        for i in range(n):
            vi1, vi2, vi3 = L[3*i,:], L[3*i+1,:], L[3*i+2,:]
            a = vi1
            b = vi2 - np.dot(vi2, a) * a
            gproj = np.dot(g, a) * a + np.dot(g, b) * b
            gproj /= norm(gproj)
            theta1 = np.arccos(np.dot(gproj, vi1))
            theta2 = np.arccos(np.dot(gproj, vi2))
            theta3 = np.arccos(np.dot(gproj, vi3))
            angle = 2 * np.pi / 3
            if theta1 <= angle and theta3 < angle:
                angles.append(theta3)
            elif theta2 < angle and theta3 <= angle:
                angles.append(2*np.pi - theta3)
            elif theta1 < angle and theta2 <= angle:
                angles.append(angle + theta1)
            else:
                raise Exception(f'{theta1} ... {theta2} ... {theta3}')
        # labels
        labels = list()
        for angle in angles:
            label = int(((angle + psi) % 2*np.pi) / (2 * np.pi / 3))
            labels.append(label)
        labels = np.array(labels)
        # sum
        s = 0
        for l in range(3):
            for i in np.argwhere(labels == l).flatten():
                for j in np.argwhere(labels != l).flatten():
                    s += W[i][j]
        sums.append(int(s/2))
    return max(sums)


if __name__ == "__main__":
    W = generate_random_graph(30)
    relax = solve_sdp_program(W)
    L = cholesky(relax)
    best_sum = find_partition(L, W)
    print(best_sum)
