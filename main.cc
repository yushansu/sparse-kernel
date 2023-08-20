#include <cassert>
#include <iostream>
#include <cstring>
#include <getopt.h>
#include <algorithm>
#include <vector>
#include <libxsmm.h>
#include <omp.h>

#include "timer.h"
#include "utils.h"
#include "csr.h"
#include "kernel.h"
#include "mmio.h"
#include "memory.h"
#include "rnd_stream.h"
#include "mkl_impl.h"

#define CHECK_ERROR_

#ifndef FWD_KB_
#define FWD_KB_ 128
#endif

#ifndef BWD_CB_
#define BWD_CB_ 128
#endif

#ifndef UPD_NB_
#define UPD_NB_ 80
#endif

#ifndef JIT_NB_
#define JIT_NB_ 16
#endif

#ifndef JIT_nb_
#define JIT_nb_ 16
#endif

#ifndef JIT_KB_
#define JIT_KB_ 128
#endif

#ifndef JIT_CB_
#define JIT_CB_ 128
#endif

#ifndef JIT_UPD_NB_
#define JIT_UPD_NB_ 16
#endif

#ifndef JIT_UPD_nb_
#define JIT_UPD_nb_ 16
#endif

#ifndef JIT_UPD_KB_
#define JIT_UPD_KB_ 32
#endif

#ifndef JIT_UPD_CB_
#define JIT_UPD_CB_ 512
#endif

using namespace std;

const struct option long_options[] = {
    {"help",           0, NULL, 'h'},
    {"version",        0, NULL, 'v'},
    {"input",          1, NULL, 'i'},
    {"batchsize",      1, NULL, 'b'},
    {"repeats",        1, NULL, 'r'},
    {"kernel",         1, NULL, 'k'},
    {NULL,             0, NULL,  0}
};

const char *const short_options = ":hvi:b:r:k:";
const char *version_info = "1.0.0";

static void Usage(const char *call) {
    std::cout << "Usage: " << call << " [OPTIONS]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "\t-h or --help         Display this information" << std::endl;
    std::cout << "\t-v or --version      Display version information" << std::endl;
    std::cout << "\t-i or --input        Input sparse tensor" << std::endl;
    std::cout << "\t-b or --batchsize    Batch size" << std::endl;
    std::cout << "\t-k or --kernel       Test kernel" << std::endl;
    std::cout << "\t-r or --repeats      Number of runs" << std::endl;
}

static void PrintVersion(const char *call) {
    std::cout << call << " version " << version_info << std::endl;
}

int main(int argc, char **argv) {
    int c = 0;
    std::string tensor_file = "input.mm";
    int batch_size = 2048;
    int repeats = 50;
    std::string kernel = "mlp";
    while ((c = getopt_long(argc, argv, short_options, long_options, NULL)) != -1) {
        switch (c) {
        case 'h':
            Usage(argv[0]);
            return 0;
        case 'v':
            PrintVersion(argv[0]);
            return 0;
        case 'i':
            tensor_file = std::string(optarg);
            break;
        case 'b':
            batch_size = atoi(optarg);
            break;
        case 'k':
            kernel = std::string(optarg);
            break;
        case 'r':
            repeats = atoi(optarg);
            break;
        case ':':
            printf("Option -%c requires an argument\n", optopt);
            return -1;
        case '?':
                printf("Unknown option `-%c'\n", optopt);
            return -1;
        default:
            Usage(argv[0]);
            return -1;
        } // switch (c) 
    } // while ((c = getopt_long(...)))
    // init matrices
    Timer *timer = detail::Singleton<Timer>::Instance();
    RndStream rnd_stream = RndStream(0);
    
    if (kernel.compare("mlp") == 0) {
        /* allocate matrices */
        // sp_W is a C * K sparse matrix, stored in the CSC format
        // sp_W_T is the CSR format
        // W, I, O are stored in column major
        CSR<int, float> sp_W(tensor_file);
        CSR<int, float>* sp_W_T = sp_W.Transpose();
        int N = batch_size;
        int K = sp_W.rows();
        int C = sp_W.cols();
        int nnz = sp_W.nnz();
        float* values = sp_W.values();
        int* colptr = sp_W.rowptr();
        int* rowidx = sp_W.colidx();
        int* rowptr = sp_W_T->rowptr();
        int* colidx = sp_W_T->colidx();
        int* cidx = rowidx;
        int* kidx = MMUtils::Alloc<int>(nnz, MMUtils::PAGE_4K, MMUtils::NODE1);
        int* idxmap = MMUtils::Alloc<int>(nnz, MMUtils::PAGE_4K, MMUtils::NODE1);
        float* W = MMUtils::Alloc<float>(C * K, MMUtils::PAGE_4K, MMUtils::NODE1);
        float* dW = MMUtils::Alloc<float>(C * K, MMUtils::PAGE_4K, MMUtils::NODE1);
        float* dvalues_ref = MMUtils::Alloc<float>(nnz, MMUtils::PAGE_4K, MMUtils::NODE1);
        float* dvalues = MMUtils::Alloc<float>(nnz, MMUtils::PAGE_4K, MMUtils::NODE1);
        float* I = MMUtils::Alloc<float>(N * C, MMUtils::PAGE_4K, MMUtils::NODE1); 
        float* O = MMUtils::Alloc<float>(N * K, MMUtils::PAGE_4K, MMUtils::NODE1);
        float* O_ref = MMUtils::Alloc<float>(N * K, MMUtils::PAGE_4K, MMUtils::NODE1);
        float* dI = MMUtils::Alloc<float>(N * C, MMUtils::PAGE_4K, MMUtils::NODE1);
        float* dI_ref = MMUtils::Alloc<float>(N * C, MMUtils::PAGE_4K, MMUtils::NODE1); 
        float* dO = MMUtils::Alloc<float>(N * K, MMUtils::PAGE_4K, MMUtils::NODE1);
        rnd_stream.Uniformfp32(-1.0, 1.0, N * C, I);
        rnd_stream.Uniformfp32(-1.0, 1.0, N * K, O);
        rnd_stream.Uniformfp32(-1.0, 1.0, N * K, dO);
        rnd_stream.Uniformfp32(-1.0, 1.0, nnz, values);
        memset(W, 0, sizeof(W[0]) * K * C);
        for (int k = 0; k < K; ++k) {
            int colstart = colptr[k];
            int colend = colptr[k + 1];
            for (int i = colstart; i < colend; i++) {
                int c = rowidx[i];
                W[k * C + c] = values[i];
            }
        }
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        descr.mode = SPARSE_FILL_MODE_LOWER;
        descr.diag = SPARSE_DIAG_NON_UNIT;
        // csr
        sparse_matrix_t csrW;
        mkl_sparse_s_create_csr(&csrW, SPARSE_INDEX_BASE_ZERO, K, C, colptr, colptr + 1, rowidx, values);
        mkl_sparse_set_mm_hint(csrW, SPARSE_OPERATION_NON_TRANSPOSE, descr,
                               SPARSE_LAYOUT_ROW_MAJOR, N, 1);
        mkl_sparse_set_memory_hint(csrW, SPARSE_MEMORY_AGGRESSIVE);
        mkl_sparse_optimize(csrW);
        // csc
        sparse_matrix_t cscW;
        mkl_sparse_s_create_csc(&cscW, SPARSE_INDEX_BASE_ZERO, C, K, colptr, colptr + 1, rowidx, values);
        mkl_sparse_set_mm_hint(cscW, SPARSE_OPERATION_NON_TRANSPOSE, descr,
                               SPARSE_LAYOUT_ROW_MAJOR, N, 1);
        mkl_sparse_set_memory_hint(cscW, SPARSE_MEMORY_AGGRESSIVE);
        mkl_sparse_optimize(cscW);
        std::cout << "N         = " << N << std::endl;
        std::cout << "C         = " << C << std::endl;
        std::cout << "K         = " << K << std::endl;
        std::cout << "nnz       = " << nnz << ", " << (1.0 - 1.0 * nnz/(C * K)) * 100.0 << "%" << std::endl;
        std::cout << "UPD_NB_   = " << UPD_NB_ << std::endl;
        std::cout << "BWD_CB_   = " << BWD_CB_ << std::endl;
        std::cout << "FWD_KB_   = " << FWD_KB_ << std::endl;
        std::cout << "JIT_NB_   = " << JIT_NB_ << std::endl;
        std::cout << "JIT_nb_   = " << JIT_nb_ << std::endl;
        std::cout << "JIT_CB_   = " << JIT_CB_ << std::endl;
        std::cout << "JIT_KB_   = " << JIT_KB_ << std::endl;
        std::cout << "JIT_U_NB_ = " << JIT_UPD_NB_ << std::endl;
        std::cout << "JIT_U_nb_ = " << JIT_UPD_nb_ << std::endl;
        std::cout << "JIT_U_CB_ = " << JIT_UPD_CB_ << std::endl;
        std::cout << "JIT_U_KB_ = " << JIT_UPD_KB_ << std::endl;
        assert(C%BWD_CB_ == 0);
        assert(K%FWD_KB_ == 0);
        assert(K%JIT_KB_ == 0);
        assert(C%JIT_CB_ == 0);
        assert(K%JIT_UPD_KB_ == 0);
        assert(C%JIT_UPD_CB_ == 0);
        assert(JIT_nb_%16 == 0);
        assert(JIT_NB_%JIT_nb_ == 0);
        assert(N%JIT_NB_ == 0);
        assert(JIT_UPD_NB_%16 == 0);
        assert(JIT_UPD_NB_%JIT_UPD_nb_ == 0);
        assert(N%JIT_UPD_NB_ == 0);
        

        std::cout << "FWD" << std::endl;
        /* FWD O = I x W */
        // MKL GEMM
        mkl_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, K, C,
                 1.0f, I, N, W, C, 0.0f, O_ref, N);
        // MKL SPMM
        memset(O, 0, sizeof(O[0]) * K * N);
        mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrW, descr,
                        SPARSE_LAYOUT_ROW_MAJOR, I, N, N, 0.0, O, N);
    #ifdef CHECK_ERROR_                
        assert( detail::CorrectnessCheck(O_ref, O, N * K, 1e-5) );
    #endif
        // SPMM
        memset(O, 0, sizeof(O[0]) * K * N);
        SpMLPFwd<FWD_KB_>(N, K, C, colptr, rowidx, values, I, O);
    #ifdef CHECK_ERROR_
        assert( detail::CorrectnessCheck(O_ref, O, N * K, 1e-5) );
    #endif
        //perf test
        for (int r = 0; r < repeats; ++r) {
            timer->Begin("MKL_DENSE_FWD");
            mkl_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, K, C,
                     1.0f, I, N, W, C, 0.0f, O, N);
            timer->End("MKL_DENSE_FWD", 2.0 * N * K * C); 
        }
        for (int r = 0; r < repeats; ++r) {
            timer->Begin("MKL_SPARSE_FWD");
            mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrW, descr,
                            SPARSE_LAYOUT_ROW_MAJOR, I, N, N, 0.0, O, N);
            timer->End("MKL_SPARSE_FWD", 2.0 * C * K * N);
        }
        for (int r = 0; r < repeats; ++r) {
            timer->Begin("OPT_FWD");
            SpMLPFwd<FWD_KB_>(N, K, C, colptr, rowidx, values, I, O);
            timer->End("OPT_FWD", 2.0 * C * K * N);
        }
        
        std::cout << "BWD" << std::endl;
        /* BWD dI = dO x W_T */
        // MKL GEMM
        mkl_gemm(CblasColMajor, CblasNoTrans, CblasTrans, N, C, K,
                 1.0f, dO, N, W, C, 0.0f, dI_ref, N);
        // MKL SPMM
        memset(dI, 0, sizeof(dI[0]) * C * N);
        mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, cscW, descr,
                        SPARSE_LAYOUT_ROW_MAJOR, dO, N, N, 0.0, dI, N);
    #ifdef CHECK_ERROR_
        assert( detail::CorrectnessCheck(dI_ref, dI, N * C, 1e-5) );
    #endif
        // SPMM
        //memset(dI, 0, sizeof(dI[0]) * C * N);
        //SpMLPBwd(N, K, C, colptr, rowidx, values, dO, dI);
        //assert( detail::CorrectnessCheck(dI_ref, dI, N * C, 1e-5) );
        memset(dI, 0, sizeof(dI[0]) * C * N);
        ComputeIdxMap(K, C, colptr, rowidx, rowptr, colidx, idxmap);
        SpMLPBwd2<BWD_CB_>(N, K, C, rowptr, colidx, idxmap, values, dO, dI);
    #ifdef CHECK_ERROR_
        assert( detail::CorrectnessCheck(dI_ref, dI, N * C, 1e-5) );
    #endif
        //perf test
        for (int r = 0; r < repeats; ++r) {
            timer->Begin("MKL_DENSE_BWD");
            mkl_gemm(CblasColMajor, CblasNoTrans, CblasTrans, N, C, K,
                     1.0f, dO, N, W, C, 0.0f, dI_ref, N);
            timer->End("MKL_DENSE_BWD", 2.0 * N * K * C); 
        }
        for (int r = 0; r < repeats; ++r) {
            timer->Begin("MKL_SPARSE_BWD");
            mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, cscW, descr,
                            SPARSE_LAYOUT_ROW_MAJOR, dO, N, N, 0.0, dI, N);
            timer->End("MKL_SPARSE_BWD", 2.0 * C * K * N);
        }
        #if 0
        for (int r = 0; r < repeats; ++r) {
            timer->Begin("OPT_BWD");
            SpMLPBwd(N, K, C, colptr, rowidx, values, dO, dI);
            timer->End("OPT_BWD", 2.0 * C * K * N);
        }
        #endif
        for (int r = 0; r < repeats; ++r) {
            timer->Begin("OPT_BWD");
            SpMLPBwd2<BWD_CB_>(N, K, C, rowptr, colidx, idxmap, values, dO, dI);
            timer->End("OPT_BWD", 2.0 * C * K * N);
        }
        
        std::cout << "UPD" << std::endl;
        /* UPDATE dW = I_T x dO */
        // MKL GEMM
        mkl_gemm(CblasColMajor, CblasTrans, CblasNoTrans, C, K, N,
                 1.0f, I, N, dO, N, 0.0f, dW, C);
        for (int k = 0; k < K; ++k) {
            int colstart = colptr[k];
            int colend = colptr[k + 1];
            for (int i = colstart; i < colend; i++) {
                int c = rowidx[i];
                dvalues_ref[i] = dW[k * C + c];
            }
        }
        // SPMM
        memset(dvalues, 0, sizeof(dvalues[0]) * nnz);
        SpMLPUpd<UPD_NB_>(N, K, C, I, dO, colptr, rowidx, dvalues);
    #ifdef CHECK_ERROR_
        assert( detail::CorrectnessCheck(dvalues_ref, dvalues, nnz, 1e-5) );
    #endif
        memset(dvalues, 0, sizeof(dvalues[0]) * nnz);
        ComputeKIdx(K, C, colptr, cidx, kidx);
        SpMLPUpd2<UPD_NB_>(N, nnz, I, dO, cidx, kidx, dvalues);
    #ifdef CHECK_ERROR_
        assert( detail::CorrectnessCheck(dvalues_ref, dvalues, nnz, 1e-5) );
    #endif
        //perf test
        for (int r = 0; r < repeats; ++r) {
            timer->Begin("MKL_DENSE_UPD");
            mkl_gemm(CblasColMajor, CblasTrans, CblasNoTrans, C, K, N,
                     1.0f, I, N, dO, N, 0.0f, dW, C);
            timer->End("MKL_DENSE_UPD", 2.0 * N * K * C); 
        }
        for (int r = 0; r < repeats; ++r) {
            timer->Begin("OPT_UPD");
            SpMLPUpd<UPD_NB_>(N, K, C, I, dO, colptr, rowidx, dvalues);
            timer->End("OPT_UPD", 2.0 * C * K * N);
        }
        for (int r = 0; r < repeats; ++r) {
            timer->Begin("OPT_UPD2");
            SpMLPUpd2<UPD_NB_>(N, nnz, I, dO, cidx, kidx, dvalues);
            timer->End("OPT_UPD2", 2.0 * C * K * N);
        }
        
        std::cout << "JIT" << std::endl;
        /* JIT FWD and BWD */
        {
            libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;\
            int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
            int nb = JIT_nb_;
            int N_NB = JIT_NB_/nb;
            unsigned num_k_blocks = K/JIT_KB_;
            unsigned num_c_blocks = C/JIT_CB_;
            int num_blocks = num_k_blocks * num_c_blocks; 
            unsigned int** b_colptr = (unsigned int**)libxsmm_aligned_malloc(num_blocks * sizeof(unsigned int*), 64);
            unsigned int** b_rowidx = (unsigned int**)libxsmm_aligned_malloc(num_blocks * sizeof(unsigned int*), 64);
            float** b_values = (float**)libxsmm_aligned_malloc(num_blocks * sizeof(float*), 64);
            int* nnzb = (int*)libxsmm_aligned_malloc(num_blocks * sizeof(int), 64);
            for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
                b_colptr[blk_idx] = (unsigned int*)libxsmm_aligned_malloc((JIT_KB_ + 1) * sizeof(unsigned int), 64);
            }
            BlockSpMatStep1(K, C, JIT_KB_, JIT_CB_, colptr, rowidx, b_colptr, nnzb);
            for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
                if (nnzb[blk_idx] != NULL) {
                    b_rowidx[blk_idx] = (unsigned int*)libxsmm_aligned_malloc(nnzb[blk_idx] * sizeof(unsigned int), 64);
                    b_values[blk_idx] = (float*)libxsmm_aligned_malloc(nnzb[blk_idx] * sizeof(float), 64);
                } else {
                    b_rowidx[blk_idx] = NULL;
                    b_values[blk_idx] = NULL;
                }
            }
            BlockSpMatStep2(K, C, JIT_KB_, JIT_CB_, colptr, rowidx, values, b_colptr, b_rowidx, b_values);        
            float* l_a = (float*)libxsmm_aligned_malloc(sizeof(float) * N * C, 64);
            float* l_b = (float*)libxsmm_aligned_malloc(sizeof(float) * C * K, 64);
            float* l_c = (float*)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
            float* l_c_gold = (float*)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
            LIBXSMM_VLA_DECL(5, float, l_p_a, l_a, C/JIT_CB_, JIT_NB_/nb, JIT_CB_, nb);
            LIBXSMM_VLA_DECL(5, float, l_p_c, l_c, K/JIT_KB_, JIT_NB_/nb, JIT_KB_, nb);
            LIBXSMM_VLA_DECL(5, float, l_p_c_gold, l_c_gold, K/JIT_KB_, JIT_NB_/nb, JIT_KB_, 16);
            for (int k = 0; k < K; ++k) {
                int colstart = colptr[k];
                int colend = colptr[k + 1];
                for (int i = colstart; i < colend; ++i) {
                    int c = rowidx[i];
                    l_b[k * C + c] = values[i];
                }
            }
            /* touch A */
            for (int l_n = 0; l_n < N/JIT_NB_; ++l_n) {
                for (int l_c = 0; l_c < C/JIT_CB_; ++l_c) {
                    for (int l_nn = 0; l_nn < JIT_NB_/nb; ++l_nn) {
                        for (int l_cc = 0; l_cc < JIT_CB_; ++l_cc) {
                            for (int l_nnn = 0; l_nnn < nb; ++l_nnn) {
                                LIBXSMM_VLA_ACCESS(5, l_p_a, l_n, l_c, l_nn, l_cc, l_nnn,
                                                C/JIT_CB_, JIT_NB_/nb, JIT_CB_, nb) = (float)libxsmm_rng_f64();
                            }
                        }
                    }
                }
            }
            /* touch C */
            for (int l_n = 0; l_n < N/JIT_NB_; ++l_n) {
                for (int l_k = 0; l_k < K/JIT_KB_; ++l_k) {
                    for (int l_nn = 0; l_nn < JIT_NB_/nb; ++l_nn) {
                        for (int l_kk = 0; l_kk < JIT_KB_; ++l_kk) {
                            for (int l_nnn = 0; l_nnn < nb; ++l_nnn) {
                                LIBXSMM_VLA_ACCESS(5, l_p_c_gold, l_n, l_k, l_nn, l_kk, l_nnn,
                                                K/JIT_KB_, JIT_NB_/nb, JIT_KB_, nb) = 0.0f;
                                LIBXSMM_VLA_ACCESS(5, l_p_c, l_n, l_k, l_nn, l_kk, l_nnn,
                                                K/JIT_KB_, JIT_NB_/nb, JIT_KB_, nb) = 0.0f;
                            }
                        }
                    }
                }
            }
            /* dense routine */
            for (int l_n = 0; l_n < N/JIT_NB_; ++l_n) {
                for (int l_k = 0; l_k < K/JIT_KB_; ++l_k) {
                    for (int l_c = 0; l_c < C/JIT_CB_; ++l_c) {
                        for (int l_nn = 0; l_nn < JIT_NB_/nb; ++l_nn) {
                            for (int l_kk = 0; l_kk < JIT_KB_; ++l_kk) {
                                int k = l_k * JIT_KB_ + l_kk;
                                for (int l_cc = 0; l_cc < JIT_CB_; ++l_cc) {
                                    int c = l_c * JIT_CB_ + l_cc;
                                    for (int l_nnn = 0; l_nnn < nb; ++l_nnn) {
                                        LIBXSMM_VLA_ACCESS(5, l_p_c_gold, l_n, l_k, l_nn, l_kk, l_nnn,
                                                        K/JIT_KB_, JIT_NB_/nb, JIT_KB_, nb) +=
                                            LIBXSMM_VLA_ACCESS(5, l_p_a, l_n, l_c, l_nn, l_cc, l_nnn,
                                                            C/JIT_CB_, JIT_NB_/nb, JIT_CB_, nb) *
                                            l_b[k * C + c];          
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // FWD
            float alpha = 1.0;
            float beta = 1.0;
            libxsmm_descriptor_blob l_xgemm_blob;
            libxsmm_gemm_descriptor** l_xgemm_desc =
                (libxsmm_gemm_descriptor**)libxsmm_aligned_malloc(num_blocks * sizeof(libxsmm_gemm_descriptor*), 64);
            libxsmm_smmfunction* mykernel =
                (libxsmm_smmfunction*)libxsmm_aligned_malloc(num_blocks * sizeof(libxsmm_smmfunction), 64);
            for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
                l_xgemm_desc[blk_idx] =
                    libxsmm_gemm_descriptor_dinit(&l_xgemm_blob, LIBXSMM_GEMM_PRECISION(float),
                                                N_NB, JIT_KB_, JIT_CB_, JIT_CB_, 0, JIT_KB_,
                                                alpha, beta, flags, prefetch);
                mykernel[blk_idx] =
                    libxsmm_create_xcsc_soa(l_xgemm_desc[blk_idx], b_colptr[blk_idx], b_rowidx[blk_idx],
                                            (const void*)b_values[blk_idx], nb).smm;
            }
            #pragma omp parallel for collapse(2)
            for (int k = 0; k < K/JIT_KB_; ++k) {
                for (int n = 0; n < N/JIT_NB_; ++n) {
                    for (int c = 0; c < C/JIT_CB_; ++c) {
                        if (b_values[k * C/JIT_CB_ + c] != NULL) {
                            mykernel[k * C/JIT_CB_ + c](&(l_a[(n * C/JIT_CB_ + c) * JIT_CB_ * JIT_NB_]),
                                                        b_values[k * C/JIT_CB_ + c],
                                                        &(l_c[(n * K/JIT_KB_ + k) * JIT_NB_ * JIT_KB_]));
                        }
                    }
                }
            }
        #ifdef CHECK_ERROR_    
            // check error
            float l_max_error = 0.0f;
            for (int i = 0; i < N * K; ++i) {
                if ( fabs(l_c[i] - l_c_gold[i]) > l_max_error ) {
                    l_max_error = (float)fabs(l_c[i] - l_c_gold[i]);
                }
            }
            std::cout << "max error = " << l_max_error << std::endl;
        #endif
            // check performace
            for (int i = 0; i < repeats; ++i) {
                timer->Begin("JIT_FWD");
                #pragma omp parallel for collapse(2)
                for (int k = 0; k < K/JIT_KB_; ++k) {
                    for (int n = 0; n < N/JIT_NB_; ++n) {
                        for (int c = 0; c < C/JIT_CB_; ++c) {
                            if (b_values[k * C/JIT_CB_ + c] != NULL) {
                                mykernel[k * C/JIT_CB_ + c](&(l_a[(n * C/JIT_CB_ + c) * JIT_CB_ * JIT_NB_]),
                                                            b_values[k * C/JIT_CB_ + c],
                                                            &(l_c[(n * K/JIT_KB_ + k) * JIT_NB_ * JIT_KB_]));
                            }
                        }
                    }
                }
                timer->End("JIT_FWD", 2.0 * N * K * C);
            }
            for (int i = 0; i < repeats; ++i) {
                timer->Begin("JIT_BWD");
                #pragma omp parallel for collapse(2)
                for (int k = 0; k < K/JIT_KB_; ++k) {
                    for (int n = 0; n < N/JIT_NB_; ++n) {
                        for (int c = 0; c < C/JIT_CB_; ++c) {
                            if (b_values[k * C/JIT_CB_ + c] != NULL) {
                                mykernel[k * C/JIT_CB_ + c](&(l_a[(n * C/JIT_CB_ + c) * JIT_CB_ * JIT_NB_]),
                                                            b_values[k * C/JIT_CB_ + c],
                                                            &(l_c[(n * K/JIT_KB_ + k) * JIT_NB_ * JIT_KB_]));
                            }
                        }
                    }
                }
                timer->End("JIT_BWD", 2.0 * N * K * C);
            }
            // clean up
            libxsmm_free(l_xgemm_desc);
            libxsmm_free(mykernel);
            libxsmm_free(l_a);
            libxsmm_free(l_c);
            for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
                libxsmm_free(b_values[blk_idx]);
                libxsmm_free(b_colptr[blk_idx]);
                libxsmm_free(b_rowidx[blk_idx]);
            }
            libxsmm_free(b_values);
            libxsmm_free(b_colptr);
            libxsmm_free(b_rowidx);
        }
        std::cout << "JIT UPD" << std::endl;
        /* JIT UPD */
        {
            libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;\
            int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
            int nb = JIT_UPD_nb_;
            int N_NB = JIT_UPD_NB_/nb;
            unsigned num_k_blocks = K/JIT_UPD_KB_;
            unsigned num_c_blocks = C/JIT_UPD_CB_;
            int num_blocks = num_k_blocks * num_c_blocks; 
            unsigned int** c_colptr = (unsigned int**)libxsmm_aligned_malloc(num_blocks * sizeof(unsigned int*), 64);
            unsigned int** c_rowidx = (unsigned int**)libxsmm_aligned_malloc(num_blocks * sizeof(unsigned int*), 64);
            float** c_values = (float**)libxsmm_aligned_malloc(num_blocks * sizeof(float*), 64);
            int* nnzc = (int*)libxsmm_aligned_malloc(num_blocks * sizeof(int), 64);
            for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
                c_colptr[blk_idx] = (unsigned int*)libxsmm_aligned_malloc((JIT_UPD_KB_ + 1) * sizeof(unsigned int), 64);
            }
            BlockSpMatStep1(K, C, JIT_UPD_KB_, JIT_UPD_CB_, colptr, rowidx, c_colptr, nnzc);
            for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
                if (nnzc[blk_idx] != 0) {
                    c_rowidx[blk_idx] = (unsigned int*)libxsmm_aligned_malloc(nnzc[blk_idx] * sizeof(unsigned int), 64);
                    c_values[blk_idx] = (float*)libxsmm_aligned_malloc(nnzc[blk_idx] * sizeof(float), 64);
                } else {
                    c_rowidx[blk_idx] = NULL;
                    c_values[blk_idx] = NULL;
                }
            }
            BlockSpMatStep2(K, C, JIT_UPD_KB_, JIT_UPD_CB_, colptr, rowidx, values, c_colptr, c_rowidx, NULL);
            for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
                if (nnzc[blk_idx] !=0 ) {
                    memset(c_values[blk_idx], 0, nnzc[blk_idx] * sizeof(float));
                }
            }
            float* l_a = (float*)libxsmm_aligned_malloc(sizeof(float) * N * C, 64);
            float* l_b = (float*)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
            float* l_c_gold = (float*)libxsmm_aligned_malloc(sizeof(float) * C * K, 64);
            LIBXSMM_VLA_DECL(5, float, l_p_a, l_a, C/JIT_UPD_CB_, JIT_UPD_NB_/nb, JIT_UPD_CB_, nb);
            LIBXSMM_VLA_DECL(5, float, l_p_b, l_b, K/JIT_UPD_KB_, JIT_UPD_NB_/nb, JIT_UPD_KB_, nb);
            /* touch A */
            for (int l_n = 0; l_n < N/JIT_UPD_NB_; ++l_n) {
                for (int l_c = 0; l_c < C/JIT_UPD_CB_; ++l_c) {
                    for (int l_nn = 0; l_nn < JIT_UPD_NB_/nb; ++l_nn) {
                        for (int l_cc = 0; l_cc < JIT_UPD_CB_; ++l_cc) {
                            for (int l_nnn = 0; l_nnn < nb; ++l_nnn) {
                                LIBXSMM_VLA_ACCESS(5, l_p_a, l_n, l_c, l_nn, l_cc, l_nnn,
                                                   C/JIT_UPD_CB_, JIT_UPD_NB_/nb, JIT_UPD_CB_, nb) =
                                    (float)libxsmm_rng_f64();
                            }
                        }
                    }
                }
            }
            /* touch B */
            for (int l_n = 0; l_n < N/JIT_UPD_NB_; ++l_n) {
                for (int l_k = 0; l_k < K/JIT_UPD_KB_; ++l_k) {
                    for (int l_nn = 0; l_nn < JIT_UPD_NB_/nb; ++l_nn) {
                        for (int l_kk = 0; l_kk < JIT_UPD_KB_; ++l_kk) {
                            for (int l_nnn = 0; l_nnn < nb; ++l_nnn) {
                                LIBXSMM_VLA_ACCESS(5, l_p_b, l_n, l_k, l_nn, l_kk, l_nnn,
                                                   K/JIT_UPD_KB_, JIT_UPD_NB_/nb, JIT_UPD_KB_, nb) =
                                    (float)libxsmm_rng_f64();
                            }
                        }
                    }
                }
            }
            /* touch C */
            for (int k = 0; k < K; ++k) {
                for (int c = 0; c < C; ++c) {
                    l_c_gold[k * C + c] = 0.0f;
                }
            }
            /* dense routine */
            for (int l_n = 0; l_n < N/JIT_UPD_NB_; ++l_n) {
                for (int l_k = 0; l_k < K/JIT_UPD_KB_; ++l_k) {
                    for (int l_c = 0; l_c < C/JIT_UPD_CB_; ++l_c) {
                        for (int l_nn = 0; l_nn < JIT_UPD_NB_/nb; ++l_nn) {
                            for (int l_kk = 0; l_kk < JIT_UPD_KB_; ++l_kk) {
                                int k = l_k * JIT_UPD_KB_ + l_kk;
                                for (int l_cc = 0; l_cc < JIT_UPD_CB_; ++l_cc) {
                                    int c = l_c * JIT_UPD_CB_ + l_cc;
                                    for (int l_nnn = 0; l_nnn < nb; ++l_nnn) {
                                         l_c_gold[k * C + c] += 
                                            LIBXSMM_VLA_ACCESS(5, l_p_a, l_n, l_c, l_nn, l_cc, l_nnn,
                                                               C/JIT_UPD_CB_, JIT_UPD_NB_/nb, JIT_UPD_CB_, nb) *
                                            LIBXSMM_VLA_ACCESS(5, l_p_b, l_n, l_k, l_nn, l_kk, l_nnn,
                                                               K/JIT_UPD_KB_, JIT_UPD_NB_/nb, JIT_UPD_KB_, nb);        
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // UPD
            float alpha = 1.0;
            float beta = 1.0;
            libxsmm_descriptor_blob l_xgemm_blob;
            libxsmm_gemm_descriptor** l_xgemm_desc =
                (libxsmm_gemm_descriptor**)libxsmm_aligned_malloc(num_blocks * sizeof(libxsmm_gemm_descriptor*), 64);
            libxsmm_smmfunction* mykernel =
                (libxsmm_smmfunction*)libxsmm_aligned_malloc(num_blocks * sizeof(libxsmm_smmfunction), 64);
            for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
                l_xgemm_desc[blk_idx] =
                    libxsmm_gemm_descriptor_dinit(&l_xgemm_blob, LIBXSMM_GEMM_PRECISION(float),
                                                JIT_UPD_CB_, JIT_UPD_KB_, N_NB,
                                                JIT_UPD_CB_, JIT_UPD_KB_, 0,
                                                alpha, beta, flags, prefetch);
                mykernel[blk_idx] =
                    libxsmm_create_xcsc_soa(l_xgemm_desc[blk_idx], c_colptr[blk_idx], c_rowidx[blk_idx],
                                            (const void*)c_values[blk_idx], nb).smm;
            }
            #pragma omp parallel for collapse(2)
            for (int k = 0; k < K/JIT_UPD_KB_; ++k) {
                for (int c = 0; c < C/JIT_UPD_CB_; ++c) {
                    for (int n = 0; n < N/JIT_UPD_NB_; ++n) {
                        if (c_values[k * C/JIT_UPD_CB_ + c] != NULL) {
                            mykernel[k * C/JIT_UPD_CB_ + c](&(l_a[(n * C/JIT_UPD_CB_ + c) * JIT_UPD_CB_ * JIT_UPD_NB_]),
                                                            &(l_b[(n * K/JIT_UPD_KB_ + k) * JIT_UPD_KB_ * JIT_UPD_NB_]),
                                                            c_values[k * C/JIT_UPD_CB_ + c]);
                        }
                    }
                }
            }
        #ifdef CHECK_ERROR_
            // check error
            float l_max_error = 0.0f;
            for (int l_k = 0; l_k < K/JIT_UPD_KB_; ++l_k) {
                for (int l_c = 0; l_c < C/JIT_UPD_CB_; ++l_c) {
                    int blk_idx = l_k * C/JIT_UPD_CB_ + l_c;
                    for (int l_kk = 0; l_kk < JIT_UPD_KB_; ++l_kk) {
                        int colstart = c_colptr[blk_idx][l_kk];
                        int colend = c_colptr[blk_idx][l_kk + 1];
                        int k = l_k * JIT_UPD_KB_ + l_kk;
                        for (int i = colstart; i < colend; ++i) {
                            int l_cc = c_rowidx[blk_idx][i];
                            int c = l_c * JIT_UPD_CB_ + l_cc;
                            float v = c_values[blk_idx][i];
                            if ( fabs(v - l_c_gold[k * C + c]) > l_max_error ) {
                                l_max_error = (float)fabs(v - l_c_gold[k * C + c]);
                            }
                        }
                    }
                }
            }
            std::cout << "max error = " << l_max_error << std::endl;
        #endif
            // check performace
            for (int i = 0; i < repeats; ++i) {
                timer->Begin("JIT_UPD");
                for (int n = 0; n < N/JIT_UPD_NB_; ++n) {
                    #pragma omp parallel for collapse(2)
                    for (int k = 0; k < K/JIT_UPD_KB_; ++k) {
                        for (int c = 0; c < C/JIT_UPD_CB_; ++c) {
                            if (c_values[k * C/JIT_UPD_CB_ + c] != NULL) {
                                mykernel[k * C/JIT_UPD_CB_ + c](&(l_a[(n * C/JIT_UPD_CB_ + c) * JIT_UPD_CB_ * JIT_UPD_NB_]),
                                                                &(l_b[(n * K/JIT_UPD_KB_ + k) * JIT_UPD_NB_ * JIT_UPD_KB_]),
                                                                c_values[k * C/JIT_UPD_CB_ + c]);
                            }
                        }
                    }
                }
                timer->End("JIT_UPD", 2.0 * N * K * C);
            }
            // clean up
            libxsmm_free(l_xgemm_desc);
            libxsmm_free(mykernel);
            libxsmm_free(l_a);
            libxsmm_free(l_b);
            for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
                libxsmm_free(c_values[blk_idx]);
                libxsmm_free(c_colptr[blk_idx]);
                libxsmm_free(c_rowidx[blk_idx]);
            }
            libxsmm_free(c_values);
            libxsmm_free(c_colptr);
            libxsmm_free(c_rowidx);
        }
        
        /* clean up */
        delete sp_W_T;
        MMUtils::Free(kidx);
        MMUtils::Free(idxmap);
        MMUtils::Free(W);
        MMUtils::Free(dW);
        MMUtils::Free(I);
        MMUtils::Free(O_ref);
        MMUtils::Free(O);
        MMUtils::Free(dO);
        MMUtils::Free(dI);
        MMUtils::Free(dI_ref);
        MMUtils::Free(dvalues);
        MMUtils::Free(dvalues_ref);
    }

    timer->Summary();
    return 0;
}
