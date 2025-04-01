import argparse
from time import time
from torch import inference_mode, empty, rand, testing, int32, float32, sparse_csr_tensor, cuda
from utilities import load_dataset, set_adjacency_matrix

import warnings
warnings.simplefilter("ignore", UserWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=["ax", "adx", "dadx"], required=True)
    parser.add_argument("--dataset", type=str, default="ca-HepPh")
    parser.add_argument("--columns", type=int, default=512, help="Overwrites default number of columns in matrix 'x'.")
    parser.add_argument("--iterations", type=int, default=50, help="Overwrites default number of matrix multiplications tests.")
    parser.add_argument("--alpha", type=int, help="Overwrites default alpha value for the adjacency matrix.")
    parser.add_argument("--atol", type=float, default=0, help="Overwrites default absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Overwrites default relative tolerance.")
    args = parser.parse_args()

    # dataset
    dataset, alpha = load_dataset(args.dataset)
    
    if args.alpha is not None:
        alpha = args.alpha
    
    # adjacency matrix
    cbm_a, _ = set_adjacency_matrix(f"cbm-{args.operation}", dataset.edge_index, alpha)
    csr_a, _ = set_adjacency_matrix(f"csr-{args.operation}", dataset.edge_index, alpha)

    del dataset.edge_index

    cbm_y = empty((dataset.num_nodes, args.columns), dtype=float32, device="cuda")
    csr_y = empty((dataset.num_nodes, args.columns), dtype=float32, device="cuda")
    pyg_y = empty((dataset.num_nodes, args.columns), dtype=float32, device="cuda")

    x = empty((dataset.num_nodes, args.columns), dtype=float32, device='cuda')
    passed_tests=0
    failed_tests=0
    print("------------------------------------------------------------")
    with inference_mode():
        for iteration in range(1, args.iterations + 1):
            x.uniform_(0,1)
            
            cuda.synchronize()

            cbm_a.matmul(x, cbm_y)

            # Call the custom kernel for matrix multiplication
            csr_a.matmul(x, csr_y)

            # Call native matrix multiplication
            pyg_y = csr_a.a @ x

            cuda.synchronize()
            # compare
            try:
                testing.assert_close(csr_y, cbm_y, atol=args.atol, rtol=args.rtol)
                testing.assert_close(csr_y, pyg_y, atol=args.atol, rtol=args.rtol)
                passed_tests += 1
            except AssertionError as e:
                failed_tests += 0

    print(f"Passed: {passed_tests} | Failed: {failed_tests}")