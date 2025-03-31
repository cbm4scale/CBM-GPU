import torch
from cbm.csr4mm import csr4mm
from cbm import csr_cusparse as cusparse

class csr4dad(csr4mm):

    def __init__(self, edge_index):
        super().__init__(edge_index)

        # represents Â.D^{⁻1/2} in cbm format 
        num_rows = self.a.size()[0]
        d = torch.zeros(num_rows,1).to(dtype=torch.float32, device="cuda")
        x = torch.ones(num_rows,1).to(dtype=torch.float32, device="cuda")

        # resort to cbm4mm to compute the outdegree
        super().matmul(x, d)

        # compute D^{⁻1/2} and flattens 
        self.D = (d ** (-1/2)).view(-1)

        # scale values
        row_indices = torch.arange(num_rows, device="cuda")
        nnz_per_row = torch.diff(self.a.crow_indices())
        nnz_row_indices = row_indices.repeat_interleave(nnz_per_row)
        new_values = self.a.values() 
        new_values *= self.D[nnz_row_indices]
        new_values *= self.D[self.a.col_indices()]
        
        self.a = torch.sparse_csr_tensor(
            self.a.crow_indices().to(dtype=torch.int32),
            self.a.col_indices().to(dtype=torch.int32),
            new_values.to(dtype=torch.float32),
            self.a.shape
        ).to(device="cuda")

        self.cusparse_extension = cusparse.cusparseCSR(self.a)

