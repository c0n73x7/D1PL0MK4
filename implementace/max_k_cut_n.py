import numpy as np
from numpy.linalg import norm, cholesky
from utils import generate_simplex
from max_k_cut_fj import solve_sdp_program


def test_graph():
    return np.array([
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]])


def find_partition_1(L, W, k):
    assert L.ndim == W.ndim == 2
    assert L.shape[0] == L.shape[1] == W.shape[0] == W.shape[1]
    L = L.copy()
    W = W.copy()
    n = L.shape[0]
    # random vectors
    g = np.random.normal(0, 1, 2*n)
    psi = np.random.uniform(0, 2*np.pi)
    # labels
    labels = list()
    for i in range(n):
        vi = L[i,:]
        ui1 = np.concatenate((vi, np.zeros(n)))
        ui2 = np.concatenate((np.zeros(n), vi))
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
    # sum
    s = 0
    for l in range(k):
        for i in np.argwhere(labels == l).flatten():
            for j in np.argwhere(labels != l).flatten():
                s += W[i][j]
    return s/2.


def find_partition_2(L, W, k):
    assert L.ndim == W.ndim == 2
    assert L.shape[0] == L.shape[1] == W.shape[0] == W.shape[1]
    L = L.copy()
    W = W.copy()
    n = L.shape[0]
    # simplex
    simplex = generate_simplex(k)
    # labels
    labels = list()
    for i in range(n):
        vi = L[i,:]
        label_vertex = np.array([np.dot(vi, np.random.normal(0, 1, n)) for _ in range(k-1)])
        distances = list()
        for j in range(k):
            simplex_vertex = simplex[j,:]
            distance_vector = label_vertex - simplex_vertex
            distances.append(norm(distance_vector))
        distances = np.array(distances)
        labels.append(np.argmin(distances))
    labels = np.array(labels)
    # sum
    s = 0
    for l in range(k):
        for i in np.argwhere(labels == l).flatten():
            for j in np.argwhere(labels != l).flatten():
                s += W[i][j]
    return s/2.


if __name__ == "__main__":
    W = test_graph()
    k = 4
    relax = solve_sdp_program(W, k)
    L = cholesky(relax)
    sums = list()
    for _ in range(1000):
        s = find_partition_1(L, W, k)
        sums.append(s)
    print(max(sums))
    sums = list()
    for _ in range(1000):
        s = find_partition_2(L, W, k)
        sums.append(s)
    print(max(sums))
