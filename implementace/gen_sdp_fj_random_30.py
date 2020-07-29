import numpy as np
from max_k_cut_fj import solve_sdp_program as sdp_fj
from utils import generate_random_graph


if __name__ == "__main__":
    n , k = 30, 3
    for density in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:        
        print(f'generate random graph with {n} vertices, and density {density}')
        W = generate_random_graph(n, density)
        print(f'solve frieze, and jerum relaxation')
        relax_fj = sdp_fj(W, k)
        path = f'./experiments/relax_fj_{n}_{density}_{k}'
        print(f'save result to "{path}"')
        np.save(path, relax_fj)
        print()
