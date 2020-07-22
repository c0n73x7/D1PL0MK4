import json
import os
import numpy as np
import numpy.linalg as la


def householder_vector(x):
    dim = x.shape[0]
    e1 = np.zeros(dim)
    e1[0] = 1    
    norm_x = la.norm(x, 2)
    q = x + np.sign(x[0])*norm_x*e1
    q = q / la.norm(q, 2)
    return q.reshape((dim, 1))


def householder_matrix(q, n):
    dim = q.shape[0]
    Q = np.eye(n)
    Q[n-dim:,n-dim:] = np.eye(dim) - 2*np.dot(q,q.T)
    return Q


def householder_qr(A):
    n, m = A.shape
    R = A
    Q = np.eye(n)
    for i in range(min(n, m)):
        q = householder_vector(R[i:,i])
        H = householder_matrix(q, n)
        Q = np.dot(Q, H.T)
        R = np.dot(H, R)
    return Q, R


def generate_simplex(k):
    v = np.ones((k, 1))
    Q, _ = householder_qr(v)
    return Q[:,1:]


def generate_random_graph(n, seed=23):
    np.random.seed(seed)
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            W[i,j] = W[j,i] = np.random.randint(2)
    return W


def load_gset_graph(path):
    # load file
    with open(path) as f:
        data = f.read().split('\n')
    # init row
    init_row = data[0].split()
    assert len(init_row) == 2
    n = int(init_row[0])
    # weighted adjacency matrix
    W = np.zeros((n,n))
    rows = [row for row in data[1:] if row != '']
    for row in rows:
        row_data = row.split()
        assert len(row_data) == 3, row_data
        i = int(row_data[0]) - 1
        j = int(row_data[1]) - 1
        wij = int(row_data[2])
        W[i,j] = W[j,i] = wij
    return W


def load_json_graph(p):
    assert os.path.exists(p), f'file "{p}" not found'
    assert os.path.splitext(p)[1] == '.json', f'"{p}" is not json'
    with open(p, 'r') as f:
        graph = json.load(f)
    n = graph.get('n')
    A = np.zeros((n, n))
    for e in graph.get('edges'):
        s, e = e[0], e[1]
        A[s,e] = A[e,s] = 1
    return A
