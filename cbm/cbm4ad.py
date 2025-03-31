import torch
from cbm.cbm4mm import cbm4mm
from cbm import cbm_cusparse as cusparse

class cbm4ad(cbm4mm):

    def __init__(self, edge_index, edge_values, alpha=0):
        super().__init__(edge_index, edge_values, alpha)

        # represents Â.D^{⁻1/2} in cbm format 
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

        self.deltas = torch.sparse_csr_tensor(
            self.deltas.crow_indices().to(dtype=torch.int32),
            self.deltas.col_indices().to(dtype=torch.int32),
            new_values.to(dtype=torch.float32),
            self.deltas.shape
        ).to(device="cuda")

        # hack to pass multipliers has an "optional argument" 
        dummy = torch.empty(0, device="cuda")

        # previous instance has no references and should be destroyed
        self.cusparse_extension = cusparse.cusparseCBM(self.deltas, 
                                                       self.mca_offset, 
                                                       self.mca_src_idx,
                                                       self.mca_dst_idx,
                                                       dummy) 
