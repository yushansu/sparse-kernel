#ifndef FUNC_H_
#define FUNC_H_

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cassert>
#include "tensor_impl.h"
#include "csr.h"
#ifdef HAVE_MKL_
#include "mkl_impl.h"
#endif

// gemm
template <typename T>
void transpose_helper(const Tensor<2, T> &A, Tensor<2, T> &B) {
    for (size_t m = 0; m < A.rows(); ++m) {
        for (size_t n = 0; n < A.cols(); ++n) {
            B(n, m) = A(m, n);
        }
    }
}

template <typename T>
double Transpose(const Tensor<2, T> &A, Tensor<2, T> &B) {
    assert(A.rows() == B.cols());
    assert(B.cols() == A.rows());
    transpose_helper(A, B);
    return 0.0;
}

// gemm
template <typename T>
void gemm_helper(const Tensor<2, T> &A, const Tensor<2, T> &B, Tensor<2, T> &C) {
    for (size_t m = 0; m < C.rows(); ++m) {
        for (size_t n = 0; n < C.cols(); ++n) {
            T tmp = T();
            for (size_t k = 0; k < A.cols(); ++k) {
                tmp += A(m, k) * B(k, n);
            }
            C(m, n) = tmp;
        }
    }    
}

template <typename T>
double Multiply(const Tensor<2, T> &A, const Tensor<2, T> &B, Tensor<2, T> &C) {
    assert(A.rows() == C.rows());
    assert(B.cols() == C.cols());
    assert(A.cols() == B.rows());
#ifdef HAVE_MKL_
    if (std::is_floating_point<T>::value) {
        mkl_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, C.rows(), C.cols(), A.cols(),
                1.0, A.data(), A.ldc(), B.data(), B.ldc(), 0.0, C.data(), C.ldc());      
    } else {
#endif
        gemm_helper(A, B, C);
#ifdef HAVE_MKL_
}
#endif
    return (2.0 * C.rows() * C.cols() * A.cols());
}

// spmm
template <typename IdxType, typename ValType>
void spmm_helper(const CSR<IdxType, ValType> &csrA,
                 const Tensor<2, ValType> &B, Tensor<2, ValType> &C) {
    size_t rowstart = 0;
    for (size_t m = 0; m < C.rows(); ++m) {
        size_t rowend = csrA.rowptr(m + 1);
        for (size_t n = 0; n < C.cols(); ++n) {
            ValType tmp = ValType();
            for (size_t j = rowstart; j < rowend; j++) {
                tmp += csrA.values(j) * B(csrA.colidx(j), n);
            }
            C(m, n) = tmp;
        }
        rowstart = rowend;
    }    
}

template <typename IdxType, typename ValType>
double Multiply(const CSR<IdxType, ValType> &csrA, const Tensor<2, ValType> &B, Tensor<2, ValType> &C) {
    assert(csrA.rows() == C.rows());
    assert(B.cols() == C.cols());
    assert(csrA.cols() == B.rows());
#ifdef HAVE_MKL_
    if (std::is_floating_point<ValType>::value) {
        mkl_spmm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA.mkl_matrix(), csrA.mkl_descr(),
                    SPARSE_LAYOUT_COLUMN_MAJOR, B.data(), B.cols(), B.ldc(),
                    0.0, C.data(), C.ldc());
    } else {
#endif
        spmm_helper(csrA, B, C);
#ifdef HAVE_MKL_
    }
#endif
    return (2.0 * csrA.nnz() * B.cols());
}

#if 0
template <typename IdxType, typename ValType>
typename std::enable_if<std::is_floating_point<ValType>::value, void>::type
Multiply(const CSR<IdxType, ValType> &csrA, const Tensor<2, ValType> &B, Tensor<2, ValType> *C) {
    assert(A.rows() == C.rows() && B.cols() == C.cols() && A.cols() == B.rows());
#ifdef HAVE_MKL_
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_LOWER;
    descr.diag = SPARSE_DIAG_NON_UNIT;
    sparse_matrix_t csrA;
    MKL_CREATE_CSR(&csrA, SPARSE_INDEX_BASE_ZERO, m, k, rowptr, rowptr + 1, colidx, values);
    sparse_matrix_t bsrA;
    mkl_sparse_convert_bsr(csrA, bs, SPARSE_LAYOUT_ROW_MAJOR, SPARSE_OPERATION_NON_TRANSPOSE, &bsrA);
    // csr
    mkl_sparse_set_mm_hint(csrA, SPARSE_OPERATION_NON_TRANSPOSE, descr,
                           SPARSE_LAYOUT_ROW_MAJOR, n, 1);
    mkl_sparse_set_memory_hint(csrA, SPARSE_MEMORY_AGGRESSIVE);
    mkl_sparse_optimize(csrA);
    FlushCache();
    unsigned long long t0 = ReadTSC();
    MKL_CSR_SPMM(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, descr,
                 SPARSE_LAYOUT_ROW_MAJOR, B, n, n, 0.0, C, n);
#else
    spmm_helper(csrA, B, C);
#endif
}
#endif

#if 0
template <typename T>
void Multiply(const Tensor<2, T> &A, const Tensor<1, T> &x, Tensor<1, T> *y) {
    assert(A.rows() == y.length() && A.cols() == x.length());
    for (size_t m = 0; m < A.rows(); ++m) {
        T tmp = T();
        for (size_t n = 0; n < A.cols(); ++n) {
            tmp += A(m, n) * x(n);
        }
        y(m) = tmp;
    }
}

template <typename IdxType, typename ValType>
void Multiply(const CSR<IdxType, ValType> &A, const Tensor<1, ValType> &x, Tensor<1, ValType> *y) {
    assert(A.rows() == y.length() && A.cols() == x.length());
    IdxType rowstart = IdxType();
    for (size_t i = 0; i < A.rows(); i++) {
        IdxType rowend = A.rowptr(i + 1);
        ValType tmp = ValType();
        for (size_t j = rowstart; j < rowend; j++) {
            tmp += A.values(j) * x(A.colidx(j));
        }
        y(i) = tmp;
    }
}
#endif

#endif // FUNC_H_