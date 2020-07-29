import numpy as np
from numpy.linalg import cholesky
from max_k_cut_fj import find_partition as find_partition_fj
from max_3_cut_gw import find_partition as find_partition_gw
from utils import generate_random_graph


if __name__ == "__main__":
    for density in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        W = generate_random_graph(30, density)
        path_fj = f'./experiments/relax_fj_30_{density}_3.npy'
        relax_fj = np.load(path_fj)
        L = cholesky(relax_fj)
        print(f'algorithm=FJ, density={density}')
        for iters in [100, 10000]:
            s = find_partition_fj(L, W, 3, iters)
            print(f'iters={iters}, sol={s}')
        # ------------------------------------------------------------
        path_gw = f'./experiments/relax_gw_30_{density}_3.npy'
        relax_gw = np.load(path_gw)
        L = cholesky(relax_gw)
        print(f'algorithm=GW, density={density}')
        for iters in [100, 10000]:
            s = find_partition_gw(L, W, iters)
            print(f'iters={iters}, sol={s}')
        print()
