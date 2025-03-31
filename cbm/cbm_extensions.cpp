#include <omp.h>
#include <immintrin.h>
#include <torch/extension.h>
#include <arbok/tarjan/tarjan_pq.h>

#include <queue>
#include <vector>
#include <numeric>
#include <algorithm>

#include "mkl.h"
#include "mkl_spblas.h"

#define CHECK_DTYPE(x, dtype) \
    TORCH_CHECK(x.scalar_type() == dtype,  \
    "\"" #x "\" is not a tensor of type \"" #dtype "\"")

static inline void check_alloc(void* ptr) {
    if (ptr == NULL) {
            std::cerr << "Memory allocation failed." << std::endl;
            exit(-1);
            // Handle allocation failure
        } else {
            // Allocation was successful, use mkl_array as needed
            std::cout << "Memory allocation successful." << std::endl;
    }
}

static inline void check_status(sparse_status_t s) {
    if (s == SPARSE_STATUS_SUCCESS) {
        return;
    }
    else if (s == SPARSE_STATUS_NOT_INITIALIZED) {
        printf("MKL: Not Initialized\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_ALLOC_FAILED) {
        printf("MKL: Not Alloc'ed\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_INVALID_VALUE) {
        printf("MKL: Invalid Value\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_EXECUTION_FAILED) {
        printf("MKL: Execution Failed\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_INTERNAL_ERROR) {
        printf("MKL: Internal Error\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_NOT_SUPPORTED) {
        printf("MKL: Not Supported\n");
        exit(-1);
    }
}


std::vector<torch::Tensor> cbm_init_(const torch::Tensor& src_row_idx,
                                     const torch::Tensor& src_col_idx,
                                     const torch::Tensor& src_values,
                                     const MKL_INT src_n_rows,
                                     const MKL_INT alpha) {
    
    CHECK_DTYPE(src_row_idx, torch::kInt32);
    CHECK_DTYPE(src_col_idx, torch::kInt32);
    CHECK_DTYPE(src_values, torch::kFloat32);

    sparse_matrix_t m_coo;
    check_status(mkl_sparse_s_create_coo(&m_coo,
                                         SPARSE_INDEX_BASE_ZERO,
                                         src_n_rows,
                                         src_n_rows,
                                         src_row_idx.size(0),
                                         src_row_idx.data_ptr<MKL_INT>(),
                                         src_col_idx.data_ptr<MKL_INT>(),
                                         src_values.data_ptr<float>()));
    
    sparse_matrix_t m_csr;
    check_status(mkl_sparse_convert_csr(m_coo, SPARSE_OPERATION_NON_TRANSPOSE, &m_csr));

    sparse_index_base_t csr_idx_t;
    MKL_INT csr_n_rows;
    MKL_INT csr_n_cols;
    MKL_INT *csr_col_idx; 
    MKL_INT *csr_row_ptr_b; 
    MKL_INT *csr_row_ptr_e;
    float *csr_values;
    
    check_status(mkl_sparse_s_export_csr(m_csr, 
                                         &csr_idx_t, 
                                         &csr_n_rows, 
                                         &csr_n_cols, 
                                         &csr_row_ptr_b, 
                                         &csr_row_ptr_e, 
                                         &csr_col_idx, 
                                         &csr_values));

    // compute syrk of dataset
    sparse_matrix_t m_syrk;
    check_status(mkl_sparse_syrk(SPARSE_OPERATION_NON_TRANSPOSE, m_csr, &m_syrk));
    
    // extract syrk in in csr format
    sparse_index_base_t syrk_idx_t;
    MKL_INT syrk_n_rows;
    MKL_INT syrk_n_cols;
    MKL_INT *syrk_col_idx; 
    MKL_INT *syrk_row_ptr_b; 
    MKL_INT *syrk_row_ptr_e;
    float *syrk_values;

    check_status(mkl_sparse_s_export_csr(m_syrk, 
                                         &syrk_idx_t, 
                                         &syrk_n_rows, 
                                         &syrk_n_cols, 
                                         &syrk_row_ptr_b, 
                                         &syrk_row_ptr_e, 
                                         &syrk_col_idx, 
                                         &syrk_values));
    
    // declare distance graph G
    std::vector<std::array<MKL_INT, 3>> G;

    MKL_INT candidates = 0;
    // populate distance graph G
    for (MKL_INT row = 0; row < syrk_n_rows; row++) {
        MKL_INT s = syrk_row_ptr_b[row]; 
        MKL_INT e = syrk_row_ptr_e[row];
        MKL_INT nnz_r1 = csr_row_ptr_e[row] - csr_row_ptr_b[row];

        G.push_back({src_n_rows, row, nnz_r1});
        candidates++;

        for (MKL_INT i = s+1; i < e; i++) {
            MKL_INT col = syrk_col_idx[i];
            MKL_INT val = syrk_values[i];
            MKL_INT nnz_r2 = csr_row_ptr_e[col] - csr_row_ptr_b[col];
            MKL_INT h = nnz_r1 + nnz_r2 - (2 * val);

            if ((h < (nnz_r1 - alpha)) && (h != nnz_r1)) {
                G.push_back({col, row, h});
                candidates++;
            }

            if ((h < (nnz_r2 - alpha)) && (h != nnz_r2)) {
                G.push_back({row, col, h});
                candidates++;
            }
        }
    }

    // initialize arbok datastructures for MSA
    arbok::TarjanPQ mca(src_n_rows+1, candidates);

    // first run to identify number of candidate edges
    // this step is required due to arbok library
    for (int row = 0; row < syrk_n_rows; row++) {
        int b = syrk_row_ptr_b[row]; 
        int e = syrk_row_ptr_e[row];
        int nnz_r1 = csr_row_ptr_e[row] - csr_row_ptr_b[row];

        // add virtual edges
        mca.create_edge(src_n_rows, row, nnz_r1);

        for (int i = b+1; i < e; i++) {
            int col = syrk_col_idx[i];
            int val = syrk_values[i];
            int nnz_r2 = csr_row_ptr_e[col] - csr_row_ptr_b[col];
            int h = nnz_r1 + nnz_r2 - (2 * val);

            // edge from v to u is a suitable candidate
            if ((h < (nnz_r1 - alpha)) && (h != nnz_r1)) {
                mca.create_edge(col, row, h);
            }

            // edge from u to v is a suitable candidate
            if ((h < (nnz_r2 - alpha)) && (h != nnz_r2)) {
                mca.create_edge(row, col, h);
            }
        }
    }

    mkl_sparse_destroy(m_coo);
    mkl_sparse_destroy(m_syrk);

    MKL_INT root = src_n_rows;
    mca.run(root);
    auto mca_list = mca.reconstruct(root);

    std::vector<std::vector<MKL_INT>> mca_adjacency(src_n_rows + 1);

    for(int edge_idx : mca_list) {
        auto [u, v, w] = G[edge_idx];
        mca_adjacency[u].push_back(v);
    }

    std::vector<float> delta_values;
    std::vector<MKL_INT> delta_row_ptr;
    std::vector<MKL_INT> delta_col_ptr;

    // probably redundant but can be improved later
    std::vector<std::vector<MKL_INT>> rooted_tree(src_n_rows + 1);

    // compute mca branches with more than 1 node
    std::vector<MKL_INT> non_empty_branches;

    for (MKL_INT v : mca_adjacency[src_n_rows])
        if (mca_adjacency[v].size() > 0) 
            non_empty_branches.push_back(v);
    
    
    // mca data to be returned
    std::vector<MKL_INT> mca_row_idx;                                       // mca row indices
    std::vector<MKL_INT> mca_col_idx;                                       // mca column indices
    std::vector<MKL_INT> mca_children = mca_adjacency[src_n_rows];          // mca root's children list for row scaling
    std::vector<MKL_INT> mca_branches(non_empty_branches.size() + 1, 0);    // mca number of non empty branches

    // intersect virtual node with its childs
    for (auto v : mca_adjacency[src_n_rows]) {
        MKL_INT v_ptr_cur = csr_row_ptr_b[v];
        MKL_INT v_ptr_max = csr_row_ptr_e[v];
 
        while(v_ptr_cur < v_ptr_max) {
            delta_col_ptr.push_back(csr_col_idx[v_ptr_cur]);
            delta_row_ptr.push_back(v);
            delta_values.push_back(1);
            v_ptr_cur++;
        }        
    }

    // traverse mca branches
    std::queue<MKL_INT> fifo;
    for (MKL_INT b = 0; b < non_empty_branches.size(); b++){
        fifo.push(non_empty_branches[b]);

        while (!fifo.empty()) {
            MKL_INT u = fifo.front(); fifo.pop();

            for (MKL_INT v : mca_adjacency[u]) {
                fifo.push(v);

                // push edge (u,v)
                mca_row_idx.push_back(u);
                mca_col_idx.push_back(v);
                mca_branches[b + 1]++;

                MKL_INT u_ptr_cur = csr_row_ptr_b[u];
                MKL_INT u_ptr_max = csr_row_ptr_e[u];
                MKL_INT v_ptr_cur = csr_row_ptr_b[v];
                MKL_INT v_ptr_max = csr_row_ptr_e[v];

                // compute row-wise intersection
                while (u_ptr_cur < u_ptr_max && v_ptr_cur < v_ptr_max) {
                    //column indices matched
                    if (csr_col_idx[u_ptr_cur] == csr_col_idx[v_ptr_cur]) {
                        u_ptr_cur++; v_ptr_cur++;
                    }
                    
                    // column index of u is larger than v's 
                    else if (csr_col_idx[u_ptr_cur] > csr_col_idx[v_ptr_cur]) {
                        delta_col_ptr.push_back(csr_col_idx[v_ptr_cur]);
                        delta_row_ptr.push_back(v);
                        delta_values.push_back(1);
                        v_ptr_cur++;
                    }

                    // column index of v is larger than u's 
                    else{
                        delta_col_ptr.push_back(csr_col_idx[u_ptr_cur]);
                        delta_row_ptr.push_back(v);
                        delta_values.push_back(-1);
                        u_ptr_cur++;
                    }
                }

                // row v doesn't have more column indices
                while(u_ptr_cur < u_ptr_max) {
                    delta_col_ptr.push_back(csr_col_idx[u_ptr_cur]);
                    delta_row_ptr.push_back(v);
                    delta_values.push_back(-1);
                    u_ptr_cur++;
                }

                // row u doesn't have more column indices
                while(v_ptr_cur < v_ptr_max){
                    delta_col_ptr.push_back(csr_col_idx[v_ptr_cur]);
                    delta_row_ptr.push_back(v);
                    delta_values.push_back(1);
                    v_ptr_cur++;
                }
            }
        }
    }

    mkl_sparse_destroy(m_csr);

    // compute prefix sum of mca_branches
    std::partial_sum(mca_branches.begin(), mca_branches.end(), mca_branches.begin());

    std::vector<torch::Tensor> result;
    result.push_back(torch::from_blob((void*)delta_row_ptr.data(), 
                                      {delta_row_ptr.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kInt32)
                                          .device(torch::kCPU))
                                      .clone());

    result.push_back(torch::from_blob((void*)delta_col_ptr.data(), 
                                      {delta_col_ptr.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kInt32)
                                          .device(torch::kCPU))
                                      .clone());

    result.push_back(torch::from_blob((void*)delta_values.data(), 
                                      {delta_values.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kFloat32)
                                          .device(torch::kCPU))
                                      .clone());

    result.push_back(torch::from_blob((void*)mca_branches.data(), 
                                      {mca_branches.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kInt32)
                                          .device(torch::kCPU))
                                      .clone());

    result.push_back(torch::from_blob((void*)mca_row_idx.data(), 
                                      {mca_row_idx.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kInt32)
                                          .device(torch::kCPU))
                                      .clone());

    result.push_back(torch::from_blob((void*)mca_col_idx.data(), 
                                      {mca_col_idx.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kInt32)
                                          .device(torch::kCPU))
                                      .clone());

    return result;
}

// naming convention threading[seq, omp] + data_type[s, d, ...] + 
// sequence_of_operations[update, spmm] + sparse_format[coo, csr] +
// mkl_interface[int32, int64] 

static inline void omp_s_spmm_update_csr_int32_(const at::Tensor& lhs_row_ptr_b, 
                                                const at::Tensor& lhs_row_ptr_e, 
                                                const at::Tensor& lhs_col_idx, 
                                                const at::Tensor& lhs_values,
                                                const at::Tensor& rhs,
                                                const at::Tensor& mca_branches, 
                                                const at::Tensor& mca_row_idx,
                                                const at::Tensor& mca_col_idx,
                                                at::Tensor& dst) {
    CHECK_DTYPE(rhs, torch::kFloat32);
    CHECK_DTYPE(dst, torch::kFloat32);
    CHECK_DTYPE(mca_branches, torch::kInt32);
    CHECK_DTYPE(mca_row_idx, torch::kInt32);
    CHECK_DTYPE(mca_col_idx, torch::kInt32);
    CHECK_DTYPE(lhs_row_ptr_b, torch::kInt32);
    CHECK_DTYPE(lhs_row_ptr_e, torch::kInt32);
    CHECK_DTYPE(lhs_col_idx, torch::kInt32);
    CHECK_DTYPE(lhs_values, torch::kFloat32);

    sparse_matrix_t m_csr; 
    check_status( mkl_sparse_s_create_csr(&m_csr, 
                                          SPARSE_INDEX_BASE_ZERO,
                                          dst.size(0), 
                                          rhs.size(1),
                                          lhs_row_ptr_b.data_ptr<MKL_INT>(), 
                                          lhs_row_ptr_e.data_ptr<MKL_INT>(),
                                          lhs_col_idx.data_ptr<MKL_INT>(),
                                          lhs_values.data_ptr<float>()));

    float *rhs_data = rhs.data_ptr<float>();
    float *dst_data = dst.data_ptr<float>();

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    check_status(mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                                 1.0f, 
                                 m_csr,
                                 descr,
                                 SPARSE_LAYOUT_ROW_MAJOR, 
                                 rhs_data, 
                                 rhs.size(1),
                                 rhs.size(1), 
                                 0.0f, 
                                 dst_data,
                                 rhs.size(1)));

    mkl_sparse_destroy(m_csr);

    MKL_INT branch_idx, edge_idx;
    MKL_INT dst_n_cols = dst.size(1);
    MKL_INT mca_num_branches = mca_branches.size(0) ;
    MKL_INT *mca_branches_data = mca_branches.data_ptr<MKL_INT>();
    MKL_INT *mca_row_idx_data = mca_row_idx.data_ptr<MKL_INT>();
    MKL_INT *mca_col_idx_data = mca_col_idx.data_ptr<MKL_INT>();

    #pragma omp parallel for schedule(dynamic, 50) private(branch_idx, edge_idx)
    for (branch_idx = 0; branch_idx < mca_num_branches - 1; branch_idx++) {
        // traverse each mca branch and update
        for (edge_idx = mca_branches_data[branch_idx]; 
             edge_idx < mca_branches_data[branch_idx + 1]; 
             edge_idx++) {
            
            float *src_row = dst_data + (dst_n_cols * mca_row_idx_data[edge_idx]);
            float *dst_row = dst_data + (dst_n_cols * mca_col_idx_data[edge_idx]);
            
            MKL_INT col_idx;
            for (col_idx = 0; col_idx < dst_n_cols - 15; col_idx += 16) {
                __m256 ymm0 = _mm256_loadu_ps(src_row + col_idx);
                __m256 ymm1 = _mm256_loadu_ps(dst_row + col_idx);
                ymm1 = _mm256_add_ps(ymm0, ymm1);
                _mm256_storeu_ps(dst_row + col_idx, ymm1);

                __m256 ymm2 = _mm256_loadu_ps(src_row + col_idx + 8);
                __m256 ymm3 = _mm256_loadu_ps(dst_row + col_idx + 8);
                ymm3 = _mm256_add_ps(ymm2, ymm3);
                _mm256_storeu_ps(dst_row + col_idx + 8, ymm3);
            }
                
            if (col_idx < dst_n_cols - 7) {
                __m256 ymm0 = _mm256_loadu_ps(src_row + col_idx);
                __m256 ymm1 = _mm256_loadu_ps(dst_row + col_idx);
                ymm1 = _mm256_add_ps(ymm0, ymm1);

                _mm256_storeu_ps(dst_row + col_idx, ymm1);
                col_idx += 8;
            }

            for (; col_idx < dst_n_cols; col_idx++) {
                dst_row[col_idx] += src_row[col_idx];
            }
        }
    }
}

static inline void omp_s_spmm_add_csr_int32_(const at::Tensor& lhs_row_ptr_b, 
                                                    const at::Tensor& lhs_row_ptr_e, 
                                                    const at::Tensor& lhs_col_idx, 
                                                    const at::Tensor& lhs_values,
                                                    const at::Tensor& rhs,
                                                    const float epsilon,
                                                    at::Tensor& dst) {
    CHECK_DTYPE(rhs, torch::kFloat32);
    CHECK_DTYPE(dst, torch::kFloat32);
    CHECK_DTYPE(lhs_row_ptr_b, torch::kInt32);
    CHECK_DTYPE(lhs_row_ptr_e, torch::kInt32);
    CHECK_DTYPE(lhs_col_idx, torch::kInt32);
    CHECK_DTYPE(lhs_values, torch::kFloat32);

    sparse_matrix_t m_csr; 
    check_status(mkl_sparse_s_create_csr(&m_csr, 
                                         SPARSE_INDEX_BASE_ZERO,
                                         dst.size(0), 
                                         rhs.size(1),
                                         lhs_row_ptr_b.data_ptr<MKL_INT>(), 
                                         lhs_row_ptr_e.data_ptr<MKL_INT>(),
                                         lhs_col_idx.data_ptr<MKL_INT>(),
                                         lhs_values.data_ptr<float>()));

    float *rhs_data = rhs.data_ptr<float>();
    float *dst_data = dst.data_ptr<float>();

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    check_status(mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                                 1.0f, 
                                 m_csr,
                                 descr,
                                 SPARSE_LAYOUT_ROW_MAJOR, 
                                 rhs_data, 
                                 rhs.size(1),
                                 rhs.size(1), 
                                 0.0f, 
                                 dst_data,
                                 rhs.size(1)));

    mkl_sparse_destroy(m_csr);

    // tread dst and rhs as vectors to compute AX + (1+epsilon)IX
    cblas_saxpy(dst.size(0) * dst.size(1), 1 + epsilon, rhs_data, 1, dst_data, 1);
}

static inline void omp_s_spmm_update_add_csr_int32_(const at::Tensor& lhs_row_ptr_b, 
                                                    const at::Tensor& lhs_row_ptr_e, 
                                                    const at::Tensor& lhs_col_idx, 
                                                    const at::Tensor& lhs_values,
                                                    const at::Tensor& rhs,
                                                    const float epsilon,
                                                    const at::Tensor& mca_branches, 
                                                    const at::Tensor& mca_row_idx,
                                                    const at::Tensor& mca_col_idx,
                                                    at::Tensor& dst) {
    CHECK_DTYPE(rhs, torch::kFloat32);
    CHECK_DTYPE(dst, torch::kFloat32);
    CHECK_DTYPE(mca_branches, torch::kInt32);
    CHECK_DTYPE(mca_row_idx, torch::kInt32);
    CHECK_DTYPE(mca_col_idx, torch::kInt32);
    CHECK_DTYPE(lhs_row_ptr_b, torch::kInt32);
    CHECK_DTYPE(lhs_row_ptr_e, torch::kInt32);
    CHECK_DTYPE(lhs_col_idx, torch::kInt32);
    CHECK_DTYPE(lhs_values, torch::kFloat32);

    sparse_matrix_t m_csr; 
    check_status(mkl_sparse_s_create_csr(&m_csr, 
                                         SPARSE_INDEX_BASE_ZERO,
                                         dst.size(0), 
                                         rhs.size(1),
                                         lhs_row_ptr_b.data_ptr<MKL_INT>(), 
                                         lhs_row_ptr_e.data_ptr<MKL_INT>(),
                                         lhs_col_idx.data_ptr<MKL_INT>(),
                                         lhs_values.data_ptr<float>()));

    float *rhs_data = rhs.data_ptr<float>();
    float *dst_data = dst.data_ptr<float>();

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    check_status(mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                                 1.0f, 
                                 m_csr,
                                 descr,
                                 SPARSE_LAYOUT_ROW_MAJOR, 
                                 rhs_data, 
                                 rhs.size(1),
                                 rhs.size(1), 
                                 0.0f, 
                                 dst_data,
                                 rhs.size(1)));

    mkl_sparse_destroy(m_csr);

    MKL_INT branch_idx, edge_idx;
    MKL_INT dst_n_cols = dst.size(1);
    MKL_INT mca_num_branches = mca_branches.size(0) ;
    MKL_INT *mca_branches_data = mca_branches.data_ptr<MKL_INT>();
    MKL_INT *mca_row_idx_data = mca_row_idx.data_ptr<MKL_INT>();
    MKL_INT *mca_col_idx_data = mca_col_idx.data_ptr<MKL_INT>();

    #pragma omp parallel for schedule(dynamic, 50) private(branch_idx, edge_idx)
    for (branch_idx = 0; branch_idx < mca_num_branches - 1; branch_idx++) {
        // traverse each mca branch and update
        for (edge_idx = mca_branches_data[branch_idx]; 
             edge_idx < mca_branches_data[branch_idx + 1]; 
             edge_idx++) {
            
            float *src_row = dst_data + (dst_n_cols * mca_row_idx_data[edge_idx]);
            float *dst_row = dst_data + (dst_n_cols * mca_col_idx_data[edge_idx]);
            
            MKL_INT col_idx;
            for (col_idx = 0; col_idx < dst_n_cols - 15; col_idx += 16) {
                __m256 ymm0 = _mm256_loadu_ps(src_row + col_idx);
                __m256 ymm1 = _mm256_loadu_ps(dst_row + col_idx);
                ymm1 = _mm256_add_ps(ymm0, ymm1);
                _mm256_storeu_ps(dst_row + col_idx, ymm1);

                __m256 ymm2 = _mm256_loadu_ps(src_row + col_idx + 8);
                __m256 ymm3 = _mm256_loadu_ps(dst_row + col_idx + 8);
                ymm3 = _mm256_add_ps(ymm2, ymm3);
                _mm256_storeu_ps(dst_row + col_idx + 8, ymm3);
            }
                
            if (col_idx < dst_n_cols - 7) {
                __m256 ymm0 = _mm256_loadu_ps(src_row + col_idx);
                __m256 ymm1 = _mm256_loadu_ps(dst_row + col_idx);
                ymm1 = _mm256_add_ps(ymm0, ymm1);

                _mm256_storeu_ps(dst_row + col_idx, ymm1);
                col_idx += 8;
            }

            for (; col_idx < dst_n_cols; col_idx++) {
                dst_row[col_idx] += src_row[col_idx];
            }
        }
    }

    // tread dst and rhs as vectors to compute AX + (1+epsilon)IX
    cblas_saxpy(dst.size(0) * dst.size(1), 1 + epsilon, rhs_data, 1, dst_data, 1);
}

static inline void omp_s_fused_spmm_update_csr_int32_(const at::Tensor& lhs_row_ptr_b, 
                                                      const at::Tensor& lhs_row_ptr_e, 
                                                      const at::Tensor& lhs_col_idx, 
                                                      const at::Tensor& lhs_values,
                                                      const at::Tensor& rhs, 
                                                      const at::Tensor& mca_branches,
                                                      const at::Tensor& mca_row_idx,
                                                      const at::Tensor& mca_col_idx,
                                                      const at::Tensor& multiplier,
                                                      at::Tensor& dst) {
    CHECK_DTYPE(rhs, torch::kFloat32);
    CHECK_DTYPE(dst, torch::kFloat32);
    CHECK_DTYPE(mca_branches, torch::kInt32);
    CHECK_DTYPE(mca_row_idx, torch::kInt32);
    CHECK_DTYPE(mca_col_idx, torch::kInt32);
    CHECK_DTYPE(lhs_row_ptr_b, torch::kInt32);
    CHECK_DTYPE(lhs_row_ptr_e, torch::kInt32);
    CHECK_DTYPE(lhs_col_idx, torch::kInt32);
    CHECK_DTYPE(lhs_values, torch::kFloat32);
    CHECK_DTYPE(multiplier, torch::kFloat32);

    sparse_matrix_t m_csr; 
    check_status( mkl_sparse_s_create_csr(&m_csr, 
                                          SPARSE_INDEX_BASE_ZERO,
                                          dst.size(0), 
                                          rhs.size(1),
                                          lhs_row_ptr_b.data_ptr<MKL_INT>(), 
                                          lhs_row_ptr_e.data_ptr<MKL_INT>(),
                                          lhs_col_idx.data_ptr<MKL_INT>(),
                                          lhs_values.data_ptr<float>()));

    float *rhs_data = rhs.data_ptr<float>();
    float *dst_data = dst.data_ptr<float>();

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    check_status(mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                                 1.0f, 
                                 m_csr,
                                 descr,
                                 SPARSE_LAYOUT_ROW_MAJOR, 
                                 rhs_data, 
                                 rhs.size(1),
                                 rhs.size(1), 
                                 0.0f, 
                                 dst_data,
                                 rhs.size(1)));

    mkl_sparse_destroy(m_csr);

    MKL_INT branch_idx, edge_idx;
    MKL_INT dst_n_cols = dst.size(1);
    MKL_INT mca_num_branches = mca_branches.size(0) ;
    MKL_INT *mca_branches_data = mca_branches.data_ptr<MKL_INT>();
    MKL_INT *mca_row_idx_data = mca_row_idx.data_ptr<MKL_INT>();
    MKL_INT *mca_col_idx_data = mca_col_idx.data_ptr<MKL_INT>();
    float* mult_data = multiplier.data_ptr<float>();

    #pragma omp parallel for schedule(dynamic, 50) private(branch_idx, edge_idx)
    for (branch_idx = 0; branch_idx < mca_num_branches - 1; branch_idx++) {
        // traverse each mca branch and update
        for (edge_idx = mca_branches_data[branch_idx]; 
             edge_idx < mca_branches_data[branch_idx + 1]; 
             edge_idx++) {
            
            MKL_INT mca_src_idx = mca_row_idx_data[edge_idx];
            MKL_INT mca_dst_idx = mca_col_idx_data[edge_idx];
            float *src_row = dst_data + (dst_n_cols * mca_src_idx);
            float *dst_row = dst_data + (dst_n_cols * mca_dst_idx);
            
            const float mult = mult_data[mca_dst_idx] / mult_data[mca_src_idx];
            const __m256 ymm_mult = _mm256_set1_ps(mult);
            
            MKL_INT col_idx;
            for (col_idx = 0; col_idx < dst_n_cols - 15; col_idx += 16) {
                __m256 ymm0 = _mm256_loadu_ps(src_row + col_idx);
                __m256 ymm1 = _mm256_loadu_ps(dst_row + col_idx);

                ymm1 = _mm256_fmadd_ps(ymm0, ymm_mult, ymm1);
                _mm256_storeu_ps(dst_row + col_idx, ymm1);

                __m256 ymm2 = _mm256_loadu_ps(src_row + col_idx + 8); 
                __m256 ymm3 = _mm256_loadu_ps(dst_row + col_idx + 8);

                ymm3 = _mm256_fmadd_ps(ymm2, ymm_mult, ymm3);
                _mm256_storeu_ps(dst_row + col_idx + 8, ymm3);
            }
                
            if (col_idx < dst_n_cols - 7) {
                __m256 ymm0 = _mm256_loadu_ps(src_row + col_idx);
                __m256 ymm1 = _mm256_loadu_ps(dst_row + col_idx);

                ymm1 = _mm256_fmadd_ps(ymm0, ymm_mult, ymm1);
                _mm256_storeu_ps(dst_row + col_idx, ymm1);

                col_idx += 8;
            }

            for (; col_idx < dst_n_cols; col_idx++) {
                dst_row[col_idx] += src_row[col_idx] * mult;
                                     
            }
        }
    }
}

static inline void omp_s_spmm_csr_int32_(const at::Tensor& lhs_row_ptr_b, 
                                         const at::Tensor& lhs_row_ptr_e, 
                                         const at::Tensor& lhs_col_idx, 
                                         const at::Tensor& lhs_values,
                                         const at::Tensor& rhs, 
                                         at::Tensor& dst) {
    CHECK_DTYPE(rhs, torch::kFloat32);
    CHECK_DTYPE(dst, torch::kFloat32);
    CHECK_DTYPE(lhs_row_ptr_b, torch::kInt32);
    CHECK_DTYPE(lhs_row_ptr_e, torch::kInt32);
    CHECK_DTYPE(lhs_col_idx, torch::kInt32);
    CHECK_DTYPE(lhs_values, torch::kFloat32);

    sparse_matrix_t m_csr; 
    check_status( mkl_sparse_s_create_csr(&m_csr, 
                                          SPARSE_INDEX_BASE_ZERO,
                                          dst.size(0), 
                                          rhs.size(0),
                                          lhs_row_ptr_b.data_ptr<MKL_INT>(), 
                                          lhs_row_ptr_e.data_ptr<MKL_INT>(),
                                          lhs_col_idx.data_ptr<MKL_INT>(),
                                          lhs_values.data_ptr<float>()));

    float *rhs_data = rhs.data_ptr<float>();
    float *dst_data = dst.data_ptr<float>();

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    check_status(mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                                 1.0f, 
                                 m_csr,
                                 descr,
                                 SPARSE_LAYOUT_ROW_MAJOR, 
                                 rhs_data, 
                                 rhs.size(1),
                                 rhs.size(1), 
                                 0.0f, 
                                 dst_data,
                                 rhs.size(1)));
}

static inline void omp_s_spmm_csr_t_int32_(const at::Tensor& lhs_row_ptr_b, 
                                           const at::Tensor& lhs_row_ptr_e, 
                                           const at::Tensor& lhs_col_idx, 
                                           const at::Tensor& lhs_values,
                                           const at::Tensor& rhs, 
                                           at::Tensor& dst) {
    CHECK_DTYPE(rhs, torch::kFloat32);
    CHECK_DTYPE(dst, torch::kFloat32);
    CHECK_DTYPE(lhs_row_ptr_b, torch::kInt32);
    CHECK_DTYPE(lhs_row_ptr_e, torch::kInt32);
    CHECK_DTYPE(lhs_col_idx, torch::kInt32);
    CHECK_DTYPE(lhs_values, torch::kFloat32);

    sparse_matrix_t m_csr; 
    check_status( mkl_sparse_s_create_csr(&m_csr, 
                                          SPARSE_INDEX_BASE_ZERO,
                                          dst.size(0), 
                                          rhs.size(0), // columns of B
                                          lhs_row_ptr_b.data_ptr<MKL_INT>(), 
                                          lhs_row_ptr_e.data_ptr<MKL_INT>(),
                                          lhs_col_idx.data_ptr<MKL_INT>(),
                                          lhs_values.data_ptr<float>()));

    float *rhs_data = rhs.data_ptr<float>();
    float *dst_data = dst.data_ptr<float>();

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    check_status(mkl_sparse_s_mm(SPARSE_OPERATION_TRANSPOSE,
                                 1.0f, 
                                 m_csr,
                                 descr,
                                 SPARSE_LAYOUT_ROW_MAJOR, 
                                 rhs_data, 
                                 rhs.size(1),
                                 rhs.size(1), 
                                 0.0f, 
                                 dst_data,
                                 rhs.size(1)));
}

static inline void omp_s_update_csr_int32_(const at::Tensor& mca_branches,
                                           const at::Tensor& mca_row_idx,
                                           const at::Tensor& mca_col_idx,
                                           at::Tensor& dst) {
    CHECK_DTYPE(dst, torch::kFloat32);
    CHECK_DTYPE(mca_branches, torch::kInt32);
    CHECK_DTYPE(mca_row_idx, torch::kInt32);
    CHECK_DTYPE(mca_col_idx, torch::kInt32);

    MKL_INT branch_idx, edge_idx;
    MKL_INT dst_n_cols = dst.size(1);
    MKL_INT mca_num_branches = mca_branches.size(0) ;
    MKL_INT *mca_branches_data = mca_branches.data_ptr<MKL_INT>();
    MKL_INT *mca_row_idx_data = mca_row_idx.data_ptr<MKL_INT>();
    MKL_INT *mca_col_idx_data = mca_col_idx.data_ptr<MKL_INT>();
    float *dst_data = dst.data_ptr<float>();

    #pragma omp parallel for schedule(dynamic, 50) private(branch_idx, edge_idx)
    for (branch_idx = 0; branch_idx < mca_num_branches - 1; branch_idx++) {
        // traverse each mca branch and update
        for (edge_idx = mca_branches_data[branch_idx]; 
             edge_idx < mca_branches_data[branch_idx + 1]; 
             edge_idx++) {
            
            float *src_row = dst_data + (dst_n_cols * mca_row_idx_data[edge_idx]);
            float *dst_row = dst_data + (dst_n_cols * mca_col_idx_data[edge_idx]);
            
            MKL_INT col_idx;
            for (col_idx = 0; col_idx < dst_n_cols - 15; col_idx += 16) {
                __m256 ymm0 = _mm256_loadu_ps(src_row + col_idx);
                __m256 ymm1 = _mm256_loadu_ps(dst_row + col_idx);
                ymm1 = _mm256_add_ps(ymm0, ymm1);
                _mm256_storeu_ps(dst_row + col_idx, ymm1);

                __m256 ymm2 = _mm256_loadu_ps(src_row + col_idx + 8);
                __m256 ymm3 = _mm256_loadu_ps(dst_row + col_idx + 8);
                ymm3 = _mm256_add_ps(ymm2, ymm3);
                _mm256_storeu_ps(dst_row + col_idx + 8, ymm3);
            }
                
            if (col_idx < dst_n_cols - 7) {
                __m256 ymm0 = _mm256_loadu_ps(src_row + col_idx);
                __m256 ymm1 = _mm256_loadu_ps(dst_row + col_idx);
                ymm1 = _mm256_add_ps(ymm0, ymm1);

                _mm256_storeu_ps(dst_row + col_idx, ymm1);
                col_idx += 8;
            }

            for (; col_idx < dst_n_cols; col_idx++) {
                dst_row[col_idx] += src_row[col_idx];
            }
        }
    }
}


static inline void omp_s_fused_update_csr_int32_(const at::Tensor& mca_branches,
                                                 const at::Tensor& mca_row_idx,
                                                 const at::Tensor& mca_col_idx,
                                                 const at::Tensor& multiplier,
                                                 at::Tensor& dst) {
    CHECK_DTYPE(dst, torch::kFloat32);
    CHECK_DTYPE(mca_branches, torch::kInt32);
    CHECK_DTYPE(mca_row_idx, torch::kInt32);
    CHECK_DTYPE(mca_col_idx, torch::kInt32);
    CHECK_DTYPE(multiplier, torch::kFloat32);

    MKL_INT branch_idx, edge_idx;
    MKL_INT dst_n_cols = dst.size(1);
    MKL_INT mca_num_branches = mca_branches.size(0) ;
    MKL_INT *mca_branches_data = mca_branches.data_ptr<MKL_INT>();
    MKL_INT *mca_row_idx_data = mca_row_idx.data_ptr<MKL_INT>();
    MKL_INT *mca_col_idx_data = mca_col_idx.data_ptr<MKL_INT>();
    float *dst_data = dst.data_ptr<float>();
    float *mult_data = multiplier.data_ptr<float>();

    #pragma omp parallel for schedule(dynamic, 50) private(branch_idx, edge_idx)
    for (branch_idx = 0; branch_idx < mca_num_branches - 1; branch_idx++) {
        // traverse each mca branch and update
        for (edge_idx = mca_branches_data[branch_idx]; 
             edge_idx < mca_branches_data[branch_idx + 1]; 
             edge_idx++) {
            
            MKL_INT mca_src_idx = mca_row_idx_data[edge_idx];
            MKL_INT mca_dst_idx = mca_col_idx_data[edge_idx];
            float *src_row = dst_data + (dst_n_cols * mca_src_idx);
            float *dst_row = dst_data + (dst_n_cols * mca_dst_idx);
            
            const float mult = mult_data[mca_dst_idx] / mult_data[mca_src_idx];
            const __m256 ymm_mult = _mm256_set1_ps(mult);
            
            MKL_INT col_idx;
            for (col_idx = 0; col_idx < dst_n_cols - 15; col_idx += 16) {
                __m256 ymm0 = _mm256_loadu_ps(src_row + col_idx);
                __m256 ymm1 = _mm256_loadu_ps(dst_row + col_idx);

                ymm1 = _mm256_fmadd_ps(ymm0, ymm_mult, ymm1);
                _mm256_storeu_ps(dst_row + col_idx, ymm1);

                __m256 ymm2 = _mm256_loadu_ps(src_row + col_idx + 8); 
                __m256 ymm3 = _mm256_loadu_ps(dst_row + col_idx + 8);

                ymm3 = _mm256_fmadd_ps(ymm2, ymm_mult, ymm3);
                _mm256_storeu_ps(dst_row + col_idx + 8, ymm3);
            }
                
            if (col_idx < dst_n_cols - 7) {
                __m256 ymm0 = _mm256_loadu_ps(src_row + col_idx);
                __m256 ymm1 = _mm256_loadu_ps(dst_row + col_idx);

                ymm1 = _mm256_fmadd_ps(ymm0, ymm_mult, ymm1);
                _mm256_storeu_ps(dst_row + col_idx, ymm1);

                col_idx += 8;
            }

            for (; col_idx < dst_n_cols; col_idx++) {
                dst_row[col_idx] += src_row[col_idx] * mult;
                                     
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &cbm_init_);
    m.def("s_spmm_update_csr_int32", &omp_s_spmm_update_csr_int32_);
    m.def("s_spmm_fused_update_csr_int32", &omp_s_fused_spmm_update_csr_int32_);
    m.def("s_spmm_add_csr_int32", &omp_s_spmm_add_csr_int32_);
    m.def("s_spmm_update_add_csr_int32", &omp_s_spmm_update_add_csr_int32_);
    m.def("s_spmm_csr_int32", &omp_s_spmm_csr_int32_);
    m.def("s_spmm_csr_t_int32", &omp_s_spmm_csr_t_int32_);
    m.def("s_update_csr_int32", &omp_s_update_csr_int32_);
    m.def("s_fused_update_csr_int32", &omp_s_fused_update_csr_int32_);
}
