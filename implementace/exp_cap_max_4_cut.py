import numpy as np
from numpy.linalg import cholesky
from max_k_cut_fj import solve_sdp_program
from cap_max_k_cut_greedy import local_search, get_sum_of_weights
from cap_max_k_cut_sdp import find_partition, balance
from utils import generate_random_graph
from tqdm import tqdm


if __name__ == "__main__":
    seq = [20, 20, 20, 20]
    k = len(seq)
    for density in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        W = generate_random_graph(30, density)
        # ------------------------------------------------------------
        sols = list()
        for _ in tqdm(range(100)):
            labels = local_search(W, seq)
            s = get_sum_of_weights(labels, W)
            sols.append(s)
        print(f'algorithm=local_search, density={density}, sol_from={min(sols)}, sol_to={max(sols)}')
        # ------------------------------------------------------------
        sols = list()
        for _ in tqdm(range(100)):
            relax = solve_sdp_program(W, k)
            L = cholesky(relax)
            res = find_partition(L, W, k)
            labels = balance(L, seq, res.get('simplex'), res.get('labels'), res.get('random_vectors'))
            s = get_sum_of_weights(labels, W)
            sols.append(s)
        print(f'algorithm=sdp, density={density}, sol_from={min(sols)}, sol_to={max(sols)}')
        print()
