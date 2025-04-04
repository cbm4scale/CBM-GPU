import argparse
from time import time
from torch import inference_mode, ones, zeros, randint, rand, arange, empty, tensor, float32, cuda
from utilities import load_dataset, print_dataset_info, set_adjacency_matrix

import warnings
warnings.simplefilter("ignore", UserWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=[
        "cbm-ax", "cbm-adx", "cbm-dadx",
        "csr-ax", "csr-adx", "csr-dadx"
    ], required=True)
    parser.add_argument("--dataset", type=str, default="ca-HepPh")
    parser.add_argument("--columns", type=int, default=512, help="Overwrites default number of columns in matrix 'x'.")
    parser.add_argument("--iterations", type=int, default=50, help="Overwrites default number of matrix multiplications tests.")
    parser.add_argument("--alpha", type=int, help="Overwrites default alpha value for the adjacency matrix.")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations.")
    args = parser.parse_args()

    # Load dataset
    dataset, alpha = load_dataset(args.dataset)
    
    if args.alpha is not None:
        alpha = args.alpha

    # dataset
    dataset, alpha = load_dataset(args.dataset)
    if args.columns:
        dataset.x = ones((dataset.num_nodes, args.columns))
        dataset.num_classes = 2
        dataset.y = zeros((dataset.num_nodes, 2))
        dataset.y[arange(dataset.num_nodes), randint(0, 2, (dataset.num_nodes,))] = 1
        print_dataset_info(f"{args.dataset}", dataset)
    
    if args.alpha is not None:
        alpha = args.alpha
    
    # adjacency matrix
    a, a_t = set_adjacency_matrix(args.operation, dataset.edge_index, alpha)
    del dataset.edge_index

    performance = []

    with inference_mode():
        x = rand((dataset.num_nodes, args.columns), dtype=float32, device='cuda')
        y = empty((dataset.num_nodes, dataset.num_features), dtype=float32, device='cuda')

        # warmup:

        # synchronize before warmup
        cuda.synchronize()
        
        for iterations in range(0, args.warmup):
            
            # matrix multiplication
            a.matmul(x, y)

        # synchronize before starting timer
        cuda.synchronize()

        time_start = time()
        
        for iterations in range(0, args.iterations):
            
            # matrix multiplication
            a.matmul(x, y)

        # synchronize before stopping timer
        cuda.synchronize()

        time_end = time()
    
    print(f"[{args.operation}] [{args.dataset}] [{alpha}] [{args.columns}]   Mean: {((time_end-time_start)/args.iterations) :.6f} s")