import torch
from cbm import cbm_mkl_cpp as cbm_mkl
from cbm import cbm_cusparse as cusparse

#TODO: remove edge_values

class cbm4mm:

    def __init__(self, edge_index, edge_values, alpha=0):
        
        # get number of rows in input dataset 
        num_rows = max(edge_index[0].max(), edge_index[1].max()) + 1

        # represent input dataset in CBM
        cbm_data = cbm_mkl.init(edge_index[0],  # row indices
                                edge_index[1],  # column indices
                                edge_values,    # value of nnz's
                                num_rows,       # number of rows
                                alpha)          # prunning param

        # unpack resulting data
        delta_edge_index = torch.stack([cbm_data[0], cbm_data[1]])
        delta_values = cbm_data[2]

        # convert matrix of deltas to COO tensor (torch.float32)
        self.deltas = torch.sparse_coo_tensor(
            delta_edge_index.to(torch.int32), 
            delta_values.to(torch.float32), 
            (num_rows, num_rows)
        ).coalesce().to_sparse_csr()

        self.deltas = torch.sparse_csr_tensor(
            self.deltas.crow_indices().to(dtype=torch.int32),
            self.deltas.col_indices().to(dtype=torch.int32),
            self.deltas.values().to(dtype=torch.float32),
            self.deltas.shape
        ).to(device="cuda")

        self.num_nodes = num_rows
        self.mca_offset = cbm_data[3].to(dtype=torch.int32, device="cuda")
        self.mca_src_idx = cbm_data[4].to(dtype=torch.int32, device="cuda")
        self.mca_dst_idx = cbm_data[5].to(dtype=torch.int32, device="cuda") 

        # hack to pass multipliers has an "optional argument" 
        dummy = torch.empty(0, device="cuda")

        self.cusparse_extension = cusparse.cusparseCBM(self.deltas, 
                                                       self.mca_offset, 
                                                       self.mca_src_idx, 
                                                       self.mca_dst_idx, 
                                                       dummy)

    def matmul(self, x, y):
        self.cusparse_extension.matmul_AX_ADX(x, y)