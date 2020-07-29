import numpy as np
from max_k_cut_fj import solve_sdp_program, test_graph
from numpy.linalg import cholesky, norm
from collections import Counter


def find_partition_2(L, W, k, iters=1000):
    assert L.ndim == W.ndim == 2
    assert L.shape[0] == L.shape[1] == W.shape[0] == W.shape[1]
    L = L.copy()
    W = W.copy()
    n = L.shape[0]
    simplex = generate_simplex(k)
    best_sums = -1
    best_labels = list()
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
        # sum
        s = 0
        for l in range(k):
            for i in np.argwhere(labels == l).flatten():
                for j in np.argwhere(labels != l).flatten():
                    s += W[i][j]
        s = int(s/2)
        if s > best_sum:
            best_sum = s
            best_labels = labels.copy()
    return {
        'sum': best_sum,
        'labels': best_labels,
        'simplex': simplex,
    }




def find_partition(L, W, iters=1000):
    assert L.ndim == W.ndim == 2
    assert L.shape[0] == L.shape[1]
    assert W.shape[0] == W.shape[1]
    assert L.shape[0] == 3*W.shape[0]
    L = L.copy()
    W = W.copy()
    n = W.shape[0]
    best_sum, best_psi = -1, -1
    best_angles, best_labels = list(), list()
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
            prod1 = np.dot(gproj, vi1)
            prod2 = np.dot(gproj, vi2)
            prod3 = np.dot(gproj, vi3)
            theta1 = np.arccos(prod1 if abs(prod1) <= 1 else np.sign(prod1))
            theta2 = np.arccos(prod2 if abs(prod2) <= 1 else np.sign(prod2))
            theta3 = np.arccos(prod3 if abs(prod3) <= 1 else np.sign(prod3))
            angle = 2 * np.pi / 3
            if theta1 <= angle and theta3 < angle:
                angles.append(theta3)
            elif theta2 < angle and theta3 <= angle:
                angles.append(2*np.pi - theta3)
            elif theta1 < angle and theta2 <= angle:
                angles.append(angle + theta1)
            else:
                raise Exception(f'{theta1} ... {theta2} ... {theta3}')
        labels = get_labels(angles, psi)
        s = get_sum_of_weights(labels, W)
        if s > best_sum:
            best_sum, best_psi = s, psi
            best_angles, best_labels = angles.copy(), labels.copy()
    return {
        'sum': best_sum,
        'psi': best_psi,
        'angles': best_angles,
        'labels': best_labels,
    }


def get_labels(angles, psi):
    labels = list()
    for angle in angles:
        label = int(((angle + psi) % 2*np.pi) / (2 * np.pi / 3))
        labels.append(label)
    return np.array(labels)


def get_sum_of_weights(labels, W):
    s = 0
    for l in range(3):
        for i in np.argwhere(labels == l).flatten():
            for j in np.argwhere(labels != l).flatten():
                s += W[i][j]
    return int(s / 2.)


def balance(seq, psi, angles, labels):
    seq, angles, labels = seq.copy(), angles.copy(), labels.copy()
    centers = [(psi + np.pi / 3) % (2*np.pi), (psi + np.pi) % (2*np.pi), (psi + 5 * np.pi / 3) % (2*np.pi)]
    filling = get_filling(seq, labels)
    while any([x > 0 for x in filling.values()]):
        label = [l for l, f in filling.items() if f > 0][0]
        label_ids = [i for i, m in enumerate(labels == label) if m]
        label_angles = np.array([angles[i] for i in label_ids])
        min_angle_idx, max_angle_idx = np.argmin(label_angles), np.argmax(label_angles)
        candidates = list()
        for free_label in [l for l, f in filling.items() if f < 0]:
            dist4min = np.abs(label_angles[min_angle_idx] - centers[free_label]) % (2*np.pi)
            dist4max = np.abs(label_angles[max_angle_idx] - centers[free_label]) % (2*np.pi)
            vertex_idx = label_ids[min_angle_idx] if dist4min < dist4max else label_ids[max_angle_idx]
            candidates.append({
                'to': free_label,
                'vertex_idx': vertex_idx,
                'dist': min(dist4min, dist4max),
            })
        best_candidate = sorted(candidates, key=lambda x: x['dist'])[0]
        labels[best_candidate['vertex_idx']] = best_candidate['to']
        filling[label] -= 1
        filling[best_candidate['to']] += 1
    return labels


def get_filling(seq, labels):
    state = dict(Counter(labels))
    state = sorted(state.items(), key = lambda x:(x[1], -x[0]) , reverse=True)
    filling = dict()
    for s, item in zip(seq, state):
        label, count = item[0], item[1]
        filling[label] = count - s
    return filling


if __name__ == "__main__":
    W = test_graph()
    seq = [5, 2, 2]
    relax = solve_sdp_program(W)
    L = cholesky(relax)
    res = find_partition(L, W)
    print(res.get('sum'))
    labels = balance(seq, res.get('psi'), res.get('angles'), res.get('labels'))
    s = get_sum_of_weights(labels, W)
    print(s)
