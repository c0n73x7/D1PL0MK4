import numpy as np
from max_3_cut_gw import solve_sdp_program as sdp_gw
from utils import generate_random_graph


if __name__ == "__main__":
    n = 30
    for density in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:        
        print(f'generate random graph with {n} vertices, and density {density}')
        W = generate_random_graph(n, density)
        print(f'solve goemans, and williamnson relaxation')
        relax_gw = sdp_gw(W)
        path = f'./experiments/relax_gw_{n}_{density}_{k}'
        print(f'save result to "{path}"')
        np.save(path, relax_gw)
        print()
