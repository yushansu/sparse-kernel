#ifndef MKL_IMPL_H_
#define MKL_IMPL_H_

#include <mkl.h>
#include <mkl_spblas.h>

void
mkl_transpose(const char ordering,
              size_t rows,
              size_t cols,
              float * AB,
              size_t lda,
              size_t ldb) {
    mkl_simatcopy(ordering, 'T', rows, cols, 1.0, AB, lda, ldb);
}

void
mkl_transpose(const char ordering,
              size_t rows,
              size_t cols,
              double * AB,
              size_t lda,
              size_t ldb) {
    mkl_dimatcopy(ordering, 'T', rows, cols, 1.0, AB, lda, ldb);
}

void
mkl_gemm(const CBLAS_LAYOUT layout,
         const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
         const MKL_INT m, const MKL_INT n, const MKL_INT k,
         const double alpha, const double *a, const MKL_INT lda,
         const double *b, const MKL_INT ldb,
         const double beta, double *c, const MKL_INT ldc) {
    cblas_dgemm(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
mkl_gemm(const CBLAS_LAYOUT layout,
         const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
         const MKL_INT m, const MKL_INT n, const MKL_INT k,
         const float alpha, const float *a, const MKL_INT lda,
         const float *b, const MKL_INT ldb,
         const float beta, float *c, const MKL_INT ldc) {
    cblas_sgemm(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void
mkl_spmm(sparse_operation_t operation, float alpha,
         const sparse_matrix_t A, struct matrix_descr descr,
         const sparse_layout_t layout, const float *x, MKL_INT columns, MKL_INT ldx,
         float beta, float *y, MKL_INT ldy) {
    mkl_sparse_s_mm(operation, alpha, A, descr,
                    layout, x, columns, ldx, 0.0, y, ldy);
}

void
mkl_spmm(sparse_operation_t operation, double alpha,
         const sparse_matrix_t A, struct matrix_descr descr,
         const sparse_layout_t layout, const double *x, MKL_INT columns, MKL_INT ldx,
         double beta, double *y, MKL_INT ldy) {
    mkl_sparse_d_mm(operation, alpha, A, descr,
                    layout, x, columns, ldx, 0.0, y, ldy);
}

void
mkl_create_csr(sparse_matrix_t *A, sparse_index_base_t indexing,
               MKL_INT rows, MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end,
               MKL_INT *col_indx, float *values) {
    mkl_sparse_s_create_csr(A, indexing, rows, cols, rows_start, rows_end, col_indx, values);
}

void
mkl_create_csr(sparse_matrix_t *A, sparse_index_base_t indexing,
               MKL_INT rows, MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end,
               MKL_INT *col_indx, double *values) {
    mkl_sparse_d_create_csr(A, indexing, rows, cols, rows_start, rows_end, col_indx, values);
}

#endif // MKL_IMPL_H_