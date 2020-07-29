import numpy as np
from numpy.linalg import cholesky
from cap_max_k_cut_greedy import local_search, get_sum_of_weights
from cap_max_3_cut import find_partition, balance
from utils import generate_random_graph


if __name__ == "__main__":
    seq = [10, 10, 10]
    for density in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        W = generate_random_graph(30, density)
        labels = local_search(W, seq)
        s = get_sum_of_weights(labels, W)
        print(f'algorithm=local_search, density={density}, sol={s}')
        # ------------------------------------------------------------
