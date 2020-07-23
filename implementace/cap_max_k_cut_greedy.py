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


def initialization(n):
    random.seed(23)
    vertices = [i for i in range(n)]
    partition = list()
    for i in range(k):
        V = list()
        while len(V) < n / k:
            item = random.choice(vertices)
            V.append(item)
            vertices.remove(item)
        partition.append(V)
    return partition


def iterative_step(k, partition, W):
    for i in range(k):
        for l in range(i+1, k):
            for u in partition[i]:
                wui = sum([W[u,x] for x in partition[i]])
                wul = sum([W[u,x] for x in partition[l]])
                for v in partition[l]:
                    wvi = sum([W[v,x] for x in partition[i]])
                    wvl = sum([W[v,x] for x in partition[l]])
                    if wui + wvl > wul + wvi:
                        return dict(i=i, l=l, u=u, v=v)
    return dict()


def local_search(W, k):
    assert W.ndim == 2
    assert W.shape[0] == W.shape[1]
    n = W.shape[0]
    assert n % k == 0
    partition = initialization(W.shape[0])
    while True:
        step = iterative_step(k, partition, W)
        if not step:
            break
        i, l = step.get('i'), step.get('l')
        u, v = step.get('u'), step.get('v')
        partition[i].append(v)
        partition[i].remove(u)
        partition[l].append(u)
        partition[l].remove(v)
    # labels
    labels = np.zeros(n) - 1
    for label, V in enumerate(partition):
        for i in V:
            labels[i] = label
    # sum
    s = 0
    for l in range(k):
        for i in np.argwhere(labels == l).flatten():
            for j in np.argwhere(labels != l).flatten():
                s += W[i][j]
    return s/2.


if __name__ == "__main__":
    k = 2
    W = test_graph()
    s = local_search(W, k)
    print(s)
