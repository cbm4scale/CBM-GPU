import torch
from cbm.cbm4mm import cbm4mm
from cbm import cbm_cusparse as cusparse

class cbm4dad(cbm4mm):

    def __init__(self, edge_index, edge_values, alpha=0):
        super().__init__(edge_index, edge_values, alpha)

        # for GCNConv Â = D^{⁻1/2} A D^{⁻1/2}, when D is the degree matrix of A 
        num_rows = self.deltas.size()[0]
        d = torch.zeros(num_rows,1).to(dtype=torch.float32, device="cuda")
        x = torch.ones(num_rows,1).to(dtype=torch.float32, device="cuda")

        # resort to cbm4mm to compute the outdegree
        super().matmul(x, d)

        # compute D^{⁻1/2} and flattens 
        self.D = (d ** (-1/2)).view(-1)

        # get csr column indices of matrix of deltas
        column_indices = self.deltas.col_indices()
        
        # scale columns of matrix of deltas
        new_values = self.deltas.values() * self.D[column_indices]

        # find nodes that are "not" in the mca
        missing_nodes = set(range(num_rows)) - set(self.deltas.col_indices())

        # scale rows that are not in the mca
        for row_idx in missing_nodes:
            row_ptr_s = self.deltas.crow_indices()[row_idx].item()
            row_ptr_e = self.deltas.crow_indices()[row_idx + 1].item()
            new_values[row_ptr_s:row_ptr_e] *= self.D[row_idx]


        self.deltas = torch.sparse_csr_tensor(
            self.deltas.crow_indices().to(dtype=torch.int32),
            self.deltas.col_indices().to(dtype=torch.int32),
            new_values.to(dtype=torch.float32),
            self.deltas.shape
        ).to(device="cuda")

        print(self.D.device)

        # previous instance has no references and should be destroyed
        self.cusparse_extension = cusparse.cusparseCBM(self.deltas,
                                                       self.mca_offset, 
                                                       self.mca_src_idx,
                                                       self.mca_dst_idx,
                                                       self.D)

    def matmul(self, x, y):
        self.cusparse_extension.matmul_DAX_DADX(x, y)