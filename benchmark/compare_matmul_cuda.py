import argparse
from time import time
from torch import inference_mode, empty, rand, testing, int32, float32, sparse_csr_tensor, cuda
from utilities import load_dataset, set_adjacency_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn", choices=["ax", "adx", "dadx"], required=True)
    parser.add_argument("--dataset", type=str, default="ca-HepPh")
    parser.add_argument("--columns", type=int, help="Number of columns to use in X. If not set, the original number of columns will be used.")
    parser.add_argument("--iterations", type=int, default=20, help="Number of matrix multiplications.")
    parser.add_argument("--alpha", type=int, help="Overwrite default alpha value for the adjacency matrix.")
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    # dataset
    dataset, alpha = load_dataset(args.dataset)
    
    if args.alpha is not None:
        alpha = args.alpha
    
    # adjacency matrix
    cbm_a, _ = set_adjacency_matrix(f"cbm-{args.nn}", dataset.edge_index, alpha)
    csr_a, _ = set_adjacency_matrix(f"csr-{args.nn}", dataset.edge_index, alpha)

    del dataset.edge_index

    cbm_y = empty((dataset.num_nodes, args.columns if args.columns else dataset.num_features), dtype=float32, device="cuda")    # this doesn't need to be done here but if we want to vary the number of columns, we need to create a new empty tensor
    csr_y = empty((dataset.num_nodes, args.columns if args.columns else dataset.num_features), dtype=float32, device="cuda")    # this doesn't need to be done here but if we want to vary the number of columns, we need to create a new empty tensor
    pyg_y = empty((dataset.num_nodes, args.columns if args.columns else dataset.num_features), dtype=float32, device="cuda")    # this doesn't need to be done here but if we want to vary the number of columns, we need to create a new empty tensor

    x = empty((dataset.num_nodes, args.columns), dtype=float32, device='cuda')

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
                print(f"[{iteration}/{args.iterations}] PASSED")
            except AssertionError as e:
                print(f"[{iteration}/{args.iterations}] FAILED: {e}")
            print("------------------------------------------------------------")

    # Check if CUDA is available
    if cuda.is_available():
        gpu_id = cuda.current_device()  # Get the active GPU ID
        total_mem = cuda.get_device_properties(gpu_id).total_memory  # Total memory in bytes
        allocated_mem = cuda.memory_allocated(gpu_id)  # Memory currently allocated by PyTorch
        cached_mem = cuda.memory_reserved(gpu_id)  # Memory reserved by PyTorch's caching allocator

        print(f"GPU: {cuda.get_device_name(gpu_id)}")
        print(f"Total Memory: {total_mem / 1024**3:.2f} GB")
        print(f"Allocated Memory: {allocated_mem / 1024**3:.2f} GB")
        print(f"Cached Memory: {cached_mem / 1024**3:.2f} GB")
    else:
        print("CUDA is not available.")
