import numpy as np
from max_3_cut_gw import solve_sdp_program, test_graph
from numpy.linalg import cholesky, norm
from collections import Counter


def find_partition(L, W, iters=1000):
    assert L.ndim == W.ndim == 2
    assert L.shape[0] == L.shape[1]
    assert W.shape[0] == W.shape[1]
    assert L.shape[0] == 3*W.shape[0]
    L = L.copy()
    W = W.copy()
    n = W.shape[0]
    best_sum = -1
    best_angles = list()
    best_labels = list()
    best_psi = -1
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
                raise Exception('angle error')
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
        if s > best_sum:
            best_sum = s
            best_psi = psi
            best_angles = angles.copy()
            best_labels = labels.copy()
    return {
        'sum': best_sum,
        'psi': best_psi,
        'angles': best_angles,
        'labels': best_labels,
    }


if __name__ == "__main__":
    W = test_graph()
    seq = [4, 3, 2]
    relax = solve_sdp_program(W)
    L = cholesky(relax)
    res = find_partition(L, W)
    angles, labels = res.get('angles'), res.get('labels')
    psi = res.get('psi')
    centers = [psi + np.pi / 3, psi + np.pi, psi + 5 * np.pi / 3]
    counter = dict(Counter(labels))
    counter = sorted(counter.items(), key = lambda x:(x[1], -x[0]) , reverse=True)

    candidates = dict()
    for s, c_item in zip(seq, counter):
        label, count = c_item[0], c_item[1]
        candidates[label] = s - count
    for label, fullnes in candidates.items():
        if fullnes > 0:
            # kandidati k presunuti
            mask = labels == label
            angles4set = list()
            for i, m in enumerate(mask):
                if m:
                    angles4set.append(angles[i])
            min_angle, max_angle = min(angles4set), max(angles4set)
            # najdi vsechny volne mnoziny
            available_sets = [k for k,v in candidates.items() if v < 0]
            for av_set in available_sets:
                print(np.abs(min_angle - centers[av_set]))
                print(np.abs(max_angle - centers[av_set]))
                # TODO: dodelat a rozdelit do funkci, je to takhle hnus
