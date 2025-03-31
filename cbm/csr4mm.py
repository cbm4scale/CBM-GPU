import torch
from cbm import csr_cusparse as cusparse

#TODO: remove edge_values

class csr4mm:

    def __init__(self, edge_index):

        # get number of rows in input dataset
        self.num_nodes = edge_index.max().item() + 1
        
        # convert edge index to csr format
        self.a = torch.sparse_coo_tensor(
            edge_index.to(torch.int32), 
            torch.ones(edge_index.size(1)).to(torch.float32), 
            (self.num_nodes, self.num_nodes)
        ).coalesce().to_sparse_csr()

        self.a = torch.sparse_csr_tensor(
            self.a.crow_indices().to(torch.int32),
            self.a.col_indices().to(torch.int32),
            self.a.values(),
            self.a.shape            
        ).to(device="cuda")

        #self.multipliers = torch.zeros(1, 1).to(dtype=torch.float, device="cuda")
        self.cusparse_extension = cusparse.cusparseCSR(self.a)


    def matmul(self, x, y):
        """
        Matrix multiplication with CSR format:

        
        Args:
            x (pytorch.Tensor): right-hand side operand matrix.
            y (pytorch.Tensor): result matrix.
        """
        self.cusparse_extension.matmul(x, y)