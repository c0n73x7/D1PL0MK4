import random
import numpy as np


def test_graph():
    return np.array([
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0]])


def initialization(n, seq):
    k = len(seq)
    labels = np.array([-1 for _ in range(n)])
    for v in range(n):
        available_labels = list()
        for label in range(k):
            if sum(labels == label) < seq[label]:
                available_labels.append(label)
        random_label = random.choice(available_labels)
        labels[v] = random_label
    return len(set(labels)), labels


def iterative_step(labels, k, W):
    labels = labels.copy()
    W = W.copy()
    for i in range(k):
        Vi = [i for i, b in enumerate(labels == i) if b]
        for l in range(i+1, k):
            Vl = [i for i, b in enumerate(labels == l) if b]
            for u in Vi:
                wui = sum([W[u,x] for x in Vi]) # W[u,u] = 0
                wul = sum([W[u,x] for x in Vl])
                for v in Vl:
                    wvi = sum([W[v,x] for x in Vi])
                    wvl = sum([W[v,x] for x in Vl]) # W[v,v] = 0
                    if wui + wvl > wul + wvi - 2 * W[u,v]:
                        labels[u], labels[v] = l, i
                        return True, labels
    return False, labels


def local_search(W, seq):
    assert W.ndim == 2
    assert W.shape[0] == W.shape[1]
    assert len(seq) > 1
    n = W.shape[0]
    k, labels = initialization(n, seq)
    while True:
        step, labels = iterative_step(labels, k, W)
        if not step:
            break
    return labels

def get_sum_of_weights(labels, W):
    k = len(set(labels))
    s = 0
    for l in range(k):
        for i in np.argwhere(labels == l).flatten():
            for j in np.argwhere(labels != l).flatten():
                s += W[i][j]
    return int(s / 2)


if __name__ == "__main__":
    W = test_graph()
    seq = [3, 2, 2]
    labels = local_search(W, seq)
    s = get_sum_of_weights(labels, W)
    print(s)
