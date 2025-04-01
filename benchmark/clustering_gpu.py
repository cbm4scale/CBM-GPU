import argparse
import time
import torch
from torch import inference_mode, cat, ones, zeros, arange, empty, tensor, sparse_coo_tensor, int32, float32, topk, unique, isin, bincount, argsort, split
from utilities import load_dataset, print_dataset_info, set_adjacency_matrix
from collections import deque
from cbm.cbm4mm import cbm4mm

import warnings
warnings.simplefilter("ignore", UserWarning)

class Stack:
    def __init__(self):
        self.stack = deque()

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        else:
            print("empty stack")

    def peek(self):
        if not self.is_empty():
            return self.stack[-1]
        return None

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ca-HepPh")
    parser.add_argument("--alpha", type=int, help="Overwrite default alpha value for the adjacency matrix.")
    parser.add_argument("--max_cluster_size", type=int, default=100000, help="Overwrite default max cluster size.")
    parser.add_argument("--hashing_functions", type=int, default=4, help="Overwrite default number of hashing functions.")
    args = parser.parse_args()

    # dataset
    dataset, alpha = load_dataset(args.dataset)
 
    print("dataset loaded...")

    if args.alpha is not None:
        alpha = args.alpha

    # 0. original matrix in COO(coalesced)
    tmp_edge_index = sparse_coo_tensor(
        dataset.edge_index.to(int32), 
        ones(dataset.edge_index.size(1)), 
        (dataset.num_nodes, dataset.num_nodes)
    ).coalesce()

    # Specify the device (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move your initial tensor to GPU
    original_edge_index = sparse_coo_tensor(
        tmp_edge_index.indices().to(torch.int32).clone().to(device),
        tmp_edge_index.values().to(torch.float32).clone().to(device),
        tmp_edge_index.shape
    ).coalesce()

    max_cluster_size = args.max_cluster_size
    num_hashes = args.hashing_functions

    # 1. create initial cluster
    initial_cluster = torch.unique(original_edge_index.indices()[0].to(device))

    # stack temporary clusters that need to be further partioned
    clusters_opened = Stack()
    clusters_opened.push(initial_cluster)

    # store final clusters
    clusters_closed = []

    start_time = time.perf_counter()

    while not clusters_opened.is_empty():
        cluster_rows = clusters_opened.pop()
        cluster_num_rows = cluster_rows.numel()

        mask = isin(original_edge_index.indices()[0], cluster_rows)
        filtered_indices = original_edge_index.indices()[:, mask]
        filtered_values = original_edge_index.values()[mask]

        # Step 4: Create the new sub-COO tensor
        edge_index = sparse_coo_tensor(
            filtered_indices, 
            filtered_values, 
            original_edge_index.shape
        ).coalesce()
        
        # get columns with highest entropy
        max_entropy = cluster_num_rows / 2
        in_degrees = ones((1, edge_index.shape[0]), dtype=float32, device=device) @ edge_index
        entropy_distance = abs(in_degrees - max_entropy)

        _, tmp_hashes = topk(entropy_distance, num_hashes, largest=False)
        hashes = tmp_hashes.sort().values

        # create operand vector
        powers_of_2 = zeros(edge_index.size(1), dtype=int32, device=device)
        powers_of_2[hashes] = 2 ** arange(num_hashes, dtype=int32, device=device)

        fingerprints = zeros(edge_index.size(0), dtype=int32, device=device)  # Initialize fingerprints
        fingerprints.scatter_add_(0, edge_index.indices()[0], powers_of_2[edge_index.indices()[1]])

        all_rows = arange(fingerprints.size(0), device=device)
        mask = isin(all_rows, cluster_rows) 
        fingerprints[~mask] = -1

        #print(f"fingerprints{fingerprints}")

        unique_fingerprints, cluster_assignment, counts = unique(fingerprints, return_inverse=True, return_counts=True)

        # fingerprints ->> select only rows in current cluster

        # Step 2: Sort inverse indices to group similar elements together
        sorted_indices = argsort(cluster_assignment)
        sorted_inverse = cluster_assignment[sorted_indices]

        # Step 3: Find group boundaries using `torch.bincount`
        counts = bincount(sorted_inverse)
        cluster_bounds = counts.cumsum(0)[:-1]  # Get split points (remove last cumulative sum)

        # Step 4: Use `torch.split` to efficiently gather indices per unique value
        if(-1 in unique_fingerprints):
            new_clusters = split(sorted_indices, counts.tolist())[1:]

        else:
            new_clusters = split(sorted_indices, counts.tolist())

        for cluster in new_clusters:
            if (len(cluster) <= max_cluster_size):
                clusters_closed.append(cluster)
            else:
                clusters_opened.push(cluster)

        del edge_index
        torch.cuda.empty_cache()
    # End the timer
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Hashing - Elapsed time: {elapsed_time:.6f} seconds")
    
    cpu_edge_index = original_edge_index.indices().cpu().to(int32)
    cpu_values = original_edge_index.values().cpu().to(float32)
    cpu_clusters = [c.cpu() for c in clusters_closed]

    cbm_input = cbm4mm(cpu_edge_index, cpu_values, alpha=0)    
    cpu_clusters = sorted(cpu_clusters, key=lambda x: x.numel())

    #print(f"#clusters: {len(cpu_clusters)}")

    merged_clusters = []

    b_ptr = 0
    e_ptr = len(cpu_clusters) - 1
    e_cluster = cpu_clusters[e_ptr]

    while b_ptr < e_ptr:
        b_cluster = cpu_clusters[b_ptr]

        if (len(b_cluster) + len(e_cluster) <= max_cluster_size):
            e_cluster = torch.cat((b_cluster, e_cluster), dim=0)
            b_ptr += 1

        else:
            merged_clusters.append(e_cluster)
            e_ptr -= 1
            e_cluster = cpu_clusters[e_ptr]

    merged_clusters.append(e_cluster)
    
    mask = isin(cpu_edge_index[0], merged_clusters[-1])
    largest_cluster_indices = cpu_edge_index[:, mask].to(int32)
    largest_cluster_values = cpu_values[mask].to(float32)
    largest_cluster_nnz = len(largest_cluster_values)

    deltas_per_cluster = []
    for i,c in enumerate(merged_clusters):
        
        mask = isin(cpu_edge_index[0], c)
        filtered_indices = cpu_edge_index[:, mask].to(int32)
        filtered_values = cpu_values[mask].to(float32)

        cbm_cluster = cbm4mm(filtered_indices, filtered_values, 0)
        deltas_per_cluster.append(len(cbm_cluster.deltas.values()))

    deltas_w_o_largest_cluster = sum(deltas_per_cluster[:-1]) + largest_cluster_nnz


    #print(f" cbm clustered compression rate: {len(cpu_values) / deltas_clusters}")

    print("\n----------------------------------------------------------------------")
    print("--------------------------Original Matrix-----------------------------")
    print("----------------------------------------------------------------------\n")
    print(f"#non-zero elements in CSR: {len(cpu_values)}")
    print(f"#non-zero elements in CBM: {len(cbm_input.deltas.values())}")
    print(f"non-zero reduction: {len(cpu_values) / len(cbm_input.deltas.values())}x")
    print("\n----------------------------------------------------------------------")
    print("--------------------------Clustered Matrix----------------------------")
    print("----------------------------------------------------------------------\n")
    print(f"#non-zero elements in CBM clusters: {sum(deltas_per_cluster)}")
    print(f"#non-zero reduction (V.S. CSR): {len(cpu_values) / sum(deltas_per_cluster)}x")
    print(f"#non-zero reduction (V.S. CBM): {len(cbm_input.deltas.values()) /  sum(deltas_per_cluster)}x")
    print("\n----------------------------------------------------------------------")
    ##print("--------------------------W/O Largest Cluster----------------------------")
    ##print("----------------------------------------------------------------------\n")
    ##print(f"#non-zero elements in CBM clusters: { deltas_w_o_largest_cluster}")
    ##print(f"#non-zero reduction (V.S. CSR): {len(cpu_values) / deltas_w_o_largest_cluster}x")
    ###print(f"#non-zero reduction (V.S. CBM): {len(cbm_input.deltas.values()) / deltas_w_o_largest_cluster}x")
    ##print("----------------------------------------------------------------------\n")
