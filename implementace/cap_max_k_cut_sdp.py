import numpy as np
from max_k_cut_fj import solve_sdp_program
from numpy.linalg import cholesky, norm
from collections import Counter
from itertools import permutations
from utils import generate_simplex


def test_graph():
    return np.array([
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0]])


def find_partition(L, W, k, iters=1000):
    assert L.ndim == W.ndim == 2
    assert L.shape[0] == L.shape[1] == W.shape[0] == W.shape[1]
    L = L.copy()
    W = W.copy()
    n = L.shape[0]
    simplex = generate_simplex(k)
    best_sum = -1
    best_labels = list()
    best_random_vectors = list()
    for _ in range(iters):
        random_vectors = [np.random.normal(0, 1, n) for _ in range(k-1)]
        labels = list()
        for i in range(n):
            vi = L[i,:]
            label_vertex = np.array([np.dot(vi, g) for g in random_vectors])
            distances = list()
            for j in range(k):
                simplex_vertex = simplex[j,:]
                distance_vector = label_vertex - simplex_vertex
                distances.append(norm(distance_vector))
            distances = np.array(distances)
            labels.append(np.argmin(distances))
        labels = np.array(labels)
        s = get_sum_of_weights(labels, W)
        if s > best_sum:
            best_sum = s
            best_labels = labels.copy()
            best_random_vectors = random_vectors.copy()
    return {
        'sum': best_sum,
        'labels': best_labels,
        'simplex': simplex,
        'random_vectors': random_vectors,
    }


def get_sum_of_weights(labels, W):
    k = len(set(labels))
    s = 0
    for l in range(k):
        for i in np.argwhere(labels == l).flatten():
            for j in np.argwhere(labels != l).flatten():
                s += W[i][j]
    return int(s/2)


def balance(L, seq, simplex, labels, random_vectors):
    seq, simplex, labels = seq.copy(), simplex.copy(), labels.copy()
    filling = get_filling(seq, labels)
    while any([x > 0 for x in filling.values()]):
        label_from = [l for l, f in filling.items() if f > 0][0]
        free_labels = [l for l, f in filling.items() if f < 0]
        label_from_ids = [i for i, m in enumerate(labels == label_from) if m]
        label_from_idx, label_to = None, None
        best_distance = 100 # init - large num
        for i in label_from_ids:
            vi = L[i,:]
            label_vertex = np.array([np.dot(vi, g) for g in random_vectors])
            distances = list()
            for j in free_labels:
                simplex_vertex = simplex[j,:]
                distance_vector = label_vertex - simplex_vertex
                distances.append(norm(distance_vector))
            distances = np.array(distances)
            if min(distances) < best_distance:
                best_distance = min(distances)
                label_from_idx = i
                label_to = free_labels[np.argmin(distances)]
        filling[label_from] -= 1
        filling[label_to] += 1
        labels[label_from_idx] = label_to
    return labels


def get_filling(seq, labels):
    counts = dict(Counter(labels))
    if len(counts) < len(seq):
        for i, _ in enumerate(seq):
            counts.setdefault(i, 0)
    best_value = -len(labels)
    best_filling = dict()
    for p_seq in permutations(seq):
        filling = dict()
        for s, item in zip(seq, counts.items()):
            label, count = item[0], item[1]
            filling[label] = count - s
        val = sum([v for v in filling.values() if v < 0])
        if best_value < val:
            best_value = val
            best_filling = filling
    return best_filling


if __name__ == "__main__":
    W = test_graph()
    seq = [3, 2, 2]
    k = len(seq)
    relax = solve_sdp_program(W, k)
    L = cholesky(relax)
    res = find_partition(L, W, k)
    labels = balance(L, seq, res.get('simplex'), res.get('labels'), res.get('random_vectors'))
    s = get_sum_of_weights(labels, W)
    print(s)
