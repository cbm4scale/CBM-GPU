#include <torch/extension.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <iostream>
#include <optional>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}

__global__ void update (float *dst,
                        const int32_t stride,
                        const int32_t *mca_branches,
                        const int32_t *mca_src_node_idx, 
                        const int32_t *mca_dst_node_idx) {
    
    // get block and thread data
    const int32_t thread_idx = threadIdx.x;
    const int32_t block_idx = blockIdx.x;
    const int32_t block_len = blockDim.x;
    const int32_t blocks_per_row = (stride + block_len - 1) / block_len;
    
    // get branch and column idx
    const int32_t branch_idx = block_idx / blocks_per_row;
    const int32_t column_idx = (block_idx % blocks_per_row) * block_len + thread_idx;

    // get branch elements based on bid and bsize
    const int32_t s_ptr = mca_branches[branch_idx];
    const int32_t e_ptr = mca_branches[branch_idx + 1];
    
    // traverse branch and update dst rows
    for (int edge = s_ptr; edge < e_ptr; edge++) {
        // avoid crossing row bounds
        if(column_idx < stride) {
            size_t src_edge = mca_src_node_idx[edge];   // row to be added
            size_t dst_edge = mca_dst_node_idx[edge];   // resulting row
            size_t src_idx = (src_edge * stride) + column_idx;
            size_t dst_idx = (dst_edge * stride) + column_idx;

            dst[dst_idx] += dst[src_idx];
        }
    }
}

//    // get block and thread id's
//    //const int32_t bid = blockIdx.x;
//    const int32_t tid = threadIdx.x;
//    // called bid for lack of a better name, matches first block of each row
//    const int32_t bid = blockIdx.x / ((stride + blockDim.x - 1) / blockDim.x);
//    
//    // block "bid" get branch "bid"
//    const int32_t s_ptr = mca_branches[bid]; //d_mca_branches[bid];
//    const int32_t e_ptr = mca_branches[bid + 1];
//
//    for (int edge = s_ptr; edge < e_ptr; edge++) {
//        int32_t src_edge = mca_src_node_idx[edge];   // row to be added
//        int32_t dst_edge = mca_dst_node_idx[edge];   // resulting row
//        int32_t src_idx = (src_edge * stride) + tid;
//        int32_t dst_idx = (dst_edge * stride) + tid;
//
//        dst[dst_idx] += dst[src_idx];
//    }

__global__ void fused_update (float *dst,
                              const int32_t stride,
                              const int32_t *mca_branches,
                              const int32_t *mca_src_node_idx, 
                              const int32_t *mca_dst_node_idx,
                              const float *multipliers) {
    // get block and thread data
    const int32_t thread_idx = threadIdx.x;
    const int32_t block_idx = blockIdx.x;
    const int32_t block_len = blockDim.x;
    const int32_t blocks_per_row = (stride + block_len - 1) / block_len;
    
    // get branch and column idx
    const int32_t branch_idx = block_idx / blocks_per_row;
    const int32_t column_idx = (block_idx % blocks_per_row) * block_len + thread_idx;

    // get branch elements based on bid and bsize
    const int32_t s_ptr = mca_branches[branch_idx];
    const int32_t e_ptr = mca_branches[branch_idx + 1];
    
    // traverse branch and update dst rows
    for (int edge = s_ptr; edge < e_ptr; edge++) {
        // avoid crossing row bounds
        if(column_idx < stride) {
            size_t src_edge = mca_src_node_idx[edge];   // row to be added
            size_t dst_edge = mca_dst_node_idx[edge];   // resulting row
            size_t src_idx = (src_edge * stride) + column_idx;
            size_t dst_idx = (dst_edge * stride) + column_idx;

            dst[dst_idx] += dst[src_idx] * multipliers[dst_edge] / multipliers[src_edge];
        }
    }
}

class cusparseCBM {
public:
    cusparseCBM(const torch::Tensor& deltas,
                const torch::Tensor& mca_branches,
                const torch::Tensor& mca_src_node_idx,
                const torch::Tensor& mca_dst_node_idx,
                const torch::Tensor& multipliers) {

        TORCH_CHECK(deltas.is_cuda(), "tensor is not a CUDA tensor");
        TORCH_CHECK(mca_branches.is_cuda(), "tensor is not a CUDA tensor");
        TORCH_CHECK(mca_src_node_idx.is_cuda(), "tensor is not a CUDA tensor");
        TORCH_CHECK(mca_dst_node_idx.is_cuda(), "tensor is not a CUDA tensor");

        // hack to pass multipliers has an optional argument
        if (multipliers.numel()) 
            TORCH_CHECK(multipliers.is_cuda(), "tensor is not a CUDA tensor");

        // keep a copy 
        tensor_ = &deltas;

        // init csr metadata
        nnz_ = deltas.values().size(0);
        num_rows_ = deltas.size(0);
        num_cols_ = deltas.size(1);

        // init csr arrays
        row_ptr_ = deltas.crow_indices().data_ptr<int32_t>();
        col_idx_ = deltas.col_indices().data_ptr<int32_t>();
        values_ = deltas.values().data_ptr<float>();

        // init cusparse
        CHECK_CUSPARSE( cusparseCreate(&handle_) )
        
        // init csr matrix with cusparse
        CHECK_CUSPARSE( cusparseCreateCsr(&descr_, num_rows_, num_cols_, nnz_,
                                          row_ptr_, col_idx_, values_,
                                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

        // init compression tree
        mca_branches_ =  mca_branches.data_ptr<int32_t>();
        mca_src_node_idx_ =  mca_src_node_idx.data_ptr<int32_t>();
        mca_dst_node_idx_ =  mca_dst_node_idx.data_ptr<int32_t>();
        mca_n_branches_ = mca_branches.size(0);

        // hack to pass multipliers has an optional argument
        if(multipliers.numel())
            multipliers_ =  multipliers.data_ptr<float>();
            
    }

    ~cusparseCBM() {
        CHECK_CUSPARSE( cusparseDestroy(handle_) )
        CHECK_CUSPARSE( cusparseDestroySpMat(descr_) )
        printf("destroying cbm instance on GPU...\n");
    }

    // matmul method for AX and ADX
    int matmul_AX_ADX(const torch::Tensor& rhs, torch::Tensor& res) {
        TORCH_CHECK(rhs.is_cuda(), "tensor is not a CUDA tensor");
        TORCH_CHECK(res.is_cuda(), "tensor is not a CUDA tensor");
        
        float *rhs_ptr = rhs.data_ptr<float>();
        float *res_ptr = res.data_ptr<float>();
        
        cusparseDnMatDescr_t rhs_descr;
        CHECK_CUSPARSE (
            cusparseCreateDnMat(&rhs_descr, num_cols_, rhs.size(1), rhs.size(1), 
                                (void *)rhs_ptr, CUDA_R_32F, CUSPARSE_ORDER_ROW)
        )
        
        cusparseDnMatDescr_t res_descr;
        CHECK_CUSPARSE (
            cusparseCreateDnMat(&res_descr, num_rows_, res.size(1), res.size(1), 
                                (void *)res_ptr, CUDA_R_32F, CUSPARSE_ORDER_ROW)
        )

        size_t bufferSize = 0;
        void *dBuffer = nullptr;
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUSPARSE (
            cusparseSpMM_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, 
                                    descr_, rhs_descr, &beta, res_descr, 
                                    CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, 
                                    &bufferSize)
        )

        // Allocate buffer
        CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

        // Perform sparse-dense matrix multiplication
        CHECK_CUSPARSE (
            cusparseSpMM(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, 
                        descr_, rhs_descr, &beta, res_descr, 
                        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer)
        )

        const int32_t block_size = 512; // TODO: tune block_size
        const int32_t blocks_per_row =  (res.size(1) + block_size - 1) / block_size;
        const int32_t n_blocks = mca_n_branches_ * blocks_per_row;

        // Update sparse-dense matrix multiplication
        update<<<n_blocks, block_size>>>(res_ptr, 
                                         res.size(1),
                                         mca_branches_,
                                         mca_src_node_idx_, 
                                         mca_dst_node_idx_);

        // Free resources
        CHECK_CUDA( cudaFree(dBuffer) )
        CHECK_CUSPARSE ( cusparseDestroyDnMat(rhs_descr) )
        CHECK_CUSPARSE ( cusparseDestroyDnMat(res_descr) )
        return 0;
    }

    // matmul method for DADX
    int matmul_DAX_DADX(const torch::Tensor& rhs, torch::Tensor& res) {
        TORCH_CHECK(rhs.is_cuda(), "tensor is not a CUDA tensor");
        TORCH_CHECK(res.is_cuda(), "tensor is not a CUDA tensor");
        
        float *rhs_ptr = rhs.data_ptr<float>();
        float *res_ptr = res.data_ptr<float>();
        
        cusparseDnMatDescr_t rhs_descr;
        CHECK_CUSPARSE (
            cusparseCreateDnMat(&rhs_descr, num_cols_, rhs.size(1), rhs.size(1), 
                                (void *)rhs_ptr, CUDA_R_32F, CUSPARSE_ORDER_ROW)
        )
        
        cusparseDnMatDescr_t res_descr;
        CHECK_CUSPARSE (
            cusparseCreateDnMat(&res_descr, num_rows_, res.size(1), res.size(1), 
                                (void *)res_ptr, CUDA_R_32F, CUSPARSE_ORDER_ROW)
        )

        size_t bufferSize = 0;
        void *dBuffer = nullptr;
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUSPARSE (
            cusparseSpMM_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, 
                                    descr_, rhs_descr, &beta, res_descr, 
                                    CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, 
                                    &bufferSize)
        )

        // Allocate buffer
        CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

        // Perform sparse-dense matrix multiplication
        CHECK_CUSPARSE (
            cusparseSpMM(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, 
                        descr_, rhs_descr, &beta, res_descr, 
                        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer)
        )

        const int32_t block_size = 512; // TODO: tune block_size
        const int32_t blocks_per_row =  (res.size(1) + block_size - 1) / block_size;
        const int32_t n_blocks = mca_n_branches_ * blocks_per_row;

        // Update sparse-dense matrix multiplication
        fused_update<<<n_blocks, block_size>>>(res_ptr, 
                                               res.size(1),
                                               mca_branches_,
                                               mca_src_node_idx_, 
                                               mca_dst_node_idx_,
                                               multipliers_);

        // Free resources
        CHECK_CUDA( cudaFree(dBuffer) )
        CHECK_CUSPARSE ( cusparseDestroyDnMat(rhs_descr) )
        CHECK_CUSPARSE ( cusparseDestroyDnMat(res_descr) )
        return 0;
    }

private:
    // keep tensor alive (is this really needed?)
    const torch::Tensor *tensor_; 

    // csr matrix metadata 
    int32_t nnz_; 
    int32_t num_rows_; 
    int32_t num_cols_;
    
    // deltas(csr) matrix coordinates
    int32_t *row_ptr_; 
    int32_t *col_idx_;

    // deltas (csr) matrix values
    float *values_;

    // compression tree data
    int32_t *mca_branches_; 
    int32_t *mca_src_node_idx_; 
    int32_t *mca_dst_node_idx_; 
    int32_t mca_n_branches_;

    // multipliers for DADX
    float *multipliers_;

    // cusparse handle
    cusparseHandle_t handle_;

    // cusparse csr descriptor
    cusparseSpMatDescr_t descr_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<cusparseCBM>(m, "cusparseCBM")
        .def(py::init<
                const torch::Tensor&, const torch::Tensor&, 
                const torch::Tensor&, const torch::Tensor&,
                const torch::Tensor&>())
        .def("matmul_AX_ADX", &cusparseCBM::matmul_AX_ADX)
        .def("matmul_DAX_DADX", &cusparseCBM::matmul_DAX_DADX);

}