import numpy as np
from numpy.linalg import cholesky
from max_k_cut_fj import find_partition as find_partition_fj
from max_k_cut_n import find_partition_1 as find_partition_n1
from max_k_cut_n import find_partition_2 as find_partition_n2
from utils import generate_random_graph


if __name__ == "__main__":
    for density in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        W = generate_random_graph(30, density)
        path = f'./experiments/relax_fj_30_{density}_3.npy'
        relax = np.load(path)
        L = cholesky(relax)
        # ------------------------------------------------------------
        print(f'algorithm=FJ, density={density}')
        for iters in [100, 10000]:
            s = find_partition_fj(L, W, 5, iters)
            print(f'iters={iters}, sol={s}')
        # ------------------------------------------------------------
        print(f'algorithm=N1, density={density}')
        for iters in [100, 10000]:
            s = find_partition_n1(L, W, 5, iters)
            print(f'iters={iters}, sol={s}')
        # ------------------------------------------------------------
        print(f'algorithm=N2, density={density}')
        for iters in [100, 10000]:
            s = find_partition_n2(L, W, 5, iters)
            print(f'iters={iters}, sol={s}')
        print()
