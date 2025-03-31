#include <torch/extension.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <iostream>

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

class cusparseCSR {
public:
    cusparseCSR(const torch::Tensor& adjacency) {

        // keep a copy 
        tensor_ = &adjacency;

        // init csr metadata
        nnz_ = adjacency.values().size(0);
        num_rows_ = adjacency.size(0);
        num_cols_ = adjacency.size(1);

        // init csr arrays
        row_ptr_ = adjacency.crow_indices().data_ptr<int32_t>();
        col_idx_ = adjacency.col_indices().data_ptr<int32_t>();
        values_ = adjacency.values().data_ptr<float>();

        // init cusparse
        CHECK_CUSPARSE( cusparseCreate(&handle_) )
        
        // init csr matrix with cusparse
        CHECK_CUSPARSE( cusparseCreateCsr(&descr_, num_rows_, num_cols_, nnz_,
                                          row_ptr_, col_idx_, values_,
                                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    }

    ~cusparseCSR() {
        CHECK_CUSPARSE( cusparseDestroy(handle_) )
        CHECK_CUSPARSE( cusparseDestroySpMat(descr_) )
        printf("destroying csr instance on GPU...\n");
    }

    int matmul(const torch::Tensor& rhs, torch::Tensor& res) {
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

    // cusparse handle
    cusparseHandle_t handle_;

    // cusparse csr descriptor
    cusparseSpMatDescr_t descr_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<cusparseCSR>(m, "cusparseCSR")
        .def(py::init<
                const torch::Tensor&>())
        .def("matmul", &cusparseCSR::matmul);
}