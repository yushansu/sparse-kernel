#ifndef KERNEL_H_
#define KERNEL_H_

#include <omp.h>
#define _m512_TYPED
#include <xmmintrin.h>
#include <immintrin.h>
#include <unordered_map>

#include "rtm.h"

// I is a N x C matrix, stored in column major format
// O is a N x K matrix, stored in column major format
// W is a C x K sparse matrix, stored in the CSC format
// O = I x W
/*
    for (int k = 0; k < K; ++k) {
        int colstart = colptr[k];
        int colend = colptr[k + 1];
        for (int n = 0; n < N; ++n) {
            O[k * N + n] = 0.0f;
        }
        for (int i = colstart; i < colend; ++i) {
            int c = rowidx[i];
            float w = values[i];
            #pragma omp simd
            #pragma unroll
            for (int n = 0; n < N; ++n) {
                O[k * N + n] += I[c * N + n] * w;
            }
        }
    }
*/
template<int KB = 256, int NB = 80>
void SpMLPFwd(int N,
              int K,
              int C,
              int *__restrict colptr,
              int *__restrict rowidx,
              float *__restrict values,
              float *__restrict I,
              float *__restrict O) {
    int N_NB = N/NB * NB;
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < K; k += KB) {
        for (int n = 0; n < N_NB; n += NB) {
            int kend = (k + KB > K ? K - k: KB);
            for (int kb = 0; kb < kend; kb++) {
                int kk = k + kb;
                int colstart = colptr[kk];
                int colend = colptr[kk + 1];
                __m512 v_O_0 = _mm512_set1_ps(0.0f);
                __m512 v_O_1 = _mm512_set1_ps(0.0f);
                __m512 v_O_2 = _mm512_set1_ps(0.0f);
                __m512 v_O_3 = _mm512_set1_ps(0.0f);
                __m512 v_O_4 = _mm512_set1_ps(0.0f);
                for (int i = colstart; i < colend; ++i) {
                    int c =  rowidx[i];
                    float w = values[i];
                    __m512 v_W = _mm512_set1_ps(w);
                    __m512 v_I_0 = _mm512_loadu_ps(&I[c * N + n +   0]);
                    __m512 v_I_1 = _mm512_loadu_ps(&I[c * N + n +  16]);
                    __m512 v_I_2 = _mm512_loadu_ps(&I[c * N + n +  32]);
                    __m512 v_I_3 = _mm512_loadu_ps(&I[c * N + n +  48]);
                    __m512 v_I_4 = _mm512_loadu_ps(&I[c * N + n +  64]);
                    v_O_0 = _mm512_fmadd_ps(v_W, v_I_0, v_O_0);
                    v_O_1 = _mm512_fmadd_ps(v_W, v_I_1, v_O_1);
                    v_O_2 = _mm512_fmadd_ps(v_W, v_I_2, v_O_2);
                    v_O_3 = _mm512_fmadd_ps(v_W, v_I_3, v_O_3);
                    v_O_4 = _mm512_fmadd_ps(v_W, v_I_4, v_O_4);
                }
                _mm512_storeu_ps(&O[kk * N + n +   0], v_O_0);
                _mm512_storeu_ps(&O[kk * N + n +  16], v_O_1);
                _mm512_storeu_ps(&O[kk * N + n +  32], v_O_2);
                _mm512_storeu_ps(&O[kk * N + n +  48], v_O_3);
                _mm512_storeu_ps(&O[kk * N + n +  64], v_O_4);
            }
        }
    }
    if (N_NB + 64 < N) {
        #pragma omp parallel for
        for (int k = 0; k < K; k += KB) {
            int kend = (k + KB > K ? K - k: KB);
            for (int kb = 0; kb < kend; kb++) {
                int kk = k + kb;
                int colstart = colptr[kk];
                int colend = colptr[kk + 1];
                __m512 v_O_0 = _mm512_set1_ps(0.0f);
                __m512 v_O_1 = _mm512_set1_ps(0.0f);
                __m512 v_O_2 = _mm512_set1_ps(0.0f);
                __m512 v_O_3 = _mm512_set1_ps(0.0f);
                for (int i = colstart; i < colend; ++i) {
                    int c =  rowidx[i];
                    float w = values[i];
                    __m512 v_W = _mm512_set1_ps(w);
                    __m512 v_I_0 = _mm512_loadu_ps(&I[c * N +   0]);
                    __m512 v_I_1 = _mm512_loadu_ps(&I[c * N +  16]);
                    __m512 v_I_2 = _mm512_loadu_ps(&I[c * N +  32]);
                    __m512 v_I_3 = _mm512_loadu_ps(&I[c * N +  48]);
                    v_O_0 = _mm512_fmadd_ps(v_W, v_I_0, v_O_0);
                    v_O_1 = _mm512_fmadd_ps(v_W, v_I_1, v_O_1);
                    v_O_2 = _mm512_fmadd_ps(v_W, v_I_2, v_O_2);
                    v_O_3 = _mm512_fmadd_ps(v_W, v_I_3, v_O_3);
                }
                _mm512_storeu_ps(&O[kk * N +   0], v_O_0);
                _mm512_storeu_ps(&O[kk * N +  16], v_O_1);
                _mm512_storeu_ps(&O[kk * N +  32], v_O_2);
                _mm512_storeu_ps(&O[kk * N +  48], v_O_3);
            }
        }
        N_NB += 64;
    }
    if (N_NB + 32 < N) {
        #pragma omp parallel for
        for (int k = 0; k < K; k++) {
            #pragma omp simd
            #pragma unroll
            for (int n = N_NB; n < N; n++) {
                O[k * N + n] = 0.0;
            }
            int colstart = colptr[k];
            int colend = colptr[k + 1];
            for (int i = colstart; i < colend; ++i) {
                int c =  rowidx[i];
                float w = values[i];
                #pragma omp simd
                #pragma unroll
                for (int n = N_NB; n < N_NB + 32; n++) {
                    O[k * N + n] += w * I[c * N + n];    
                }
            }
        }
        N_NB += 32;
    }
    if (N_NB + 16 < N) {
        #pragma omp parallel for
        for (int k = 0; k < K; k++) {
            #pragma omp simd
            #pragma unroll
            for (int n = N_NB; n < N; n++) {
                O[k * N + n] = 0.0;
            }
            int colstart = colptr[k];
            int colend = colptr[k + 1];
            for (int i = colstart; i < colend; ++i) {
                int c =  rowidx[i];
                float w = values[i];
                #pragma omp simd
                #pragma unroll
                for (int n = N_NB; n < N_NB + 16; n++) {
                    O[k * N + n] += w * I[c * N + n];    
                }
            }
        }
        N_NB += 16;
    }
    if (N_NB < N) {
        #pragma omp parallel for
        for (int k = 0; k < K; k++) {
            #pragma omp simd
            #pragma unroll
            for (int n = N_NB; n < N; n++) {
                O[k * N + n] = 0.0;
            }
            int colstart = colptr[k];
            int colend = colptr[k + 1];
            for (int i = colstart; i < colend; ++i) {
                int c =  rowidx[i];
                float w = values[i];
                #pragma omp simd
                #pragma unroll
                for (int n = N_NB; n < N; n++) {
                    O[k * N + n] += w * I[c * N + n];    
                }
            }
        }
    }
}

// dO is a N x K matrix, stored in column major format
// W is a C x K sparse matrix, stored in the CSC format
// dI is a N x C matrix, stored in column major format
// dI = dO x W_T
/*
    for (int c = 0; c < C; ++c) {
        for (int n = 0; n < N; ++n) {
            dI[c * N + n] = 0.0f;
        }
    }
    for (int k = 0; k < K; ++k) {
        int colstart = colptr[k];
        int colend = colptr[k + 1];
        for (int i = colstart; i < colend; ++i) {
            int c = rowidx[i];
            float w = values[i];
            for (int n = 0; n < N; ++n) {
                dI[c * N + n] += dO[k * N + n] * w;
            }
        }
    }
*/
void SpMLPBwd(int N,
              int K,
              int C,
              int *__restrict colptr,
              int *__restrict rowidx,
              float *__restrict values,
              float *__restrict dO,
              float *__restrict dI) {
    #pragma omp parallel for
    for (int c = 0; c < C; ++c) {
        #pragma omp simd
        for (int n = 0; n < N; ++n) {
            dI[c * N + n] = 0.0f;
        }
    }

    SimpleSpinLock fallback_lock;
    #pragma omp parallel for
    for (int k = 0; k < K; ++k) {
        int colstart = colptr[k];
        int colend = colptr[k + 1];
        for (int i = colstart; i < colend; ++i) {
            int c = rowidx[i];
            float w = values[i];
            {
                TransactionScope guard(fallback_lock, 100, omp_get_thread_num());
                #pragma omp simd
                for (int n = 0; n < N; ++n) {
                    dI[c * N + n] += dO[k * N + n] * w;
                }
            }
        }
    }
}

void ComputeIdxMap(int K,
                   int C,
                   int *__restrict colptr,
                   int *__restrict rowidx,
                   int *__restrict rowptr,
                   int *__restrict colidx,
                   int *__restrict idxmap) {
    std::unordered_map<int, int> hash_map;
    for (int k = 0; k < K; ++k) {
        int colstart = colptr[k];
        int colend = colptr[k + 1];
        for (int i = colstart; i < colend; ++i) {
            int c = rowidx[i];
            int dense_idx = k * C + c;
            hash_map[dense_idx] = i;
        }
    }

    for (int c = 0; c < C; ++c) {
        int rowstart = rowptr[c];
        int rowend = rowptr[c + 1];
        for (int i = rowstart; i < rowend; ++i) {
            int k = colidx[i];
            int dense_idx = k * C + c;
            idxmap[i] = hash_map[dense_idx];
        }
    }
}

// dO is a N x K matrix, stored in column major format
// W is a C x K sparse matrix, stored in the CSR format
// dI is a N x C matrix, stored in column major format
// dI = dO x W_T
/*
    for (int c = 0; c < C; ++c) {
        int rowstart = rowptr[c];
        int rowend = rowptr[c + 1];
        for (int n = 0; n < N; ++n) {
            dI[c * N + n] = 0.0f;
        }
        for (int i = rowstart; i < rowend; ++i) {
            int k = colidx[i];
            float w = values[idxmap[i]];
            for (int n = 0; n < N; ++n) {
                dI[c * N + n] += dO[k * N + n] * w;
            }
        }
    }
*/
template<int CB = 256,int NB = 80>
void SpMLPBwd2(int N,
               int K,
               int C,
               int *__restrict rowptr,
               int *__restrict colidx,
               int *__restrict idxmap,
               float *__restrict values,
               float *__restrict dO,
               float *__restrict dI) {
    int N_NB = N/NB * NB;
    #pragma omp parallel for collapse(2)
    for (int c = 0; c < C; c += CB) {
        for (int n = 0; n < N_NB; n += NB) {
            int cend = (c + CB > C ? C - c: CB);
            for (int cb = 0; cb < cend; cb++) {
                int cc = c + cb;
                int rowstart = rowptr[cc];
                int rowend = rowptr[cc + 1];
                __m512 v_dI_0 = _mm512_set1_ps(0.0f);
                __m512 v_dI_1 = _mm512_set1_ps(0.0f);
                __m512 v_dI_2 = _mm512_set1_ps(0.0f);
                __m512 v_dI_3 = _mm512_set1_ps(0.0f);
                __m512 v_dI_4 = _mm512_set1_ps(0.0f);
                for (int i = rowstart; i < rowend; ++i) {
                    int k =  colidx[i];
                    float w = values[idxmap[i]];
                    __m512 v_W = _mm512_set1_ps(w);
                    __m512 v_dO_0 = _mm512_loadu_ps(&dO[k * N + n]);
                    __m512 v_dO_1 = _mm512_loadu_ps(&dO[k * N + n + 16]);
                    __m512 v_dO_2 = _mm512_loadu_ps(&dO[k * N + n + 32]);
                    __m512 v_dO_3 = _mm512_loadu_ps(&dO[k * N + n + 48]);
                    __m512 v_dO_4 = _mm512_loadu_ps(&dO[k * N + n + 64]);
                    v_dI_0 = _mm512_fmadd_ps(v_W, v_dO_0, v_dI_0);
                    v_dI_1 = _mm512_fmadd_ps(v_W, v_dO_1, v_dI_1);
                    v_dI_2 = _mm512_fmadd_ps(v_W, v_dO_2, v_dI_2);
                    v_dI_3 = _mm512_fmadd_ps(v_W, v_dO_3, v_dI_3);
                    v_dI_4 = _mm512_fmadd_ps(v_W, v_dO_4, v_dI_4);
                }
                _mm512_storeu_ps(&dI[cc * N + n], v_dI_0);
                _mm512_storeu_ps(&dI[cc * N + n + 16], v_dI_1);
                _mm512_storeu_ps(&dI[cc * N + n + 32], v_dI_2);
                _mm512_storeu_ps(&dI[cc * N + n + 48], v_dI_3);
                _mm512_storeu_ps(&dI[cc * N + n + 64], v_dI_4);

            }
        }
    }
    if (N_NB < N) {
        #pragma omp parallel for
        for (int c = 0; c < C; c++) {
            #pragma omp simd
            #pragma unroll
            for (int n = N_NB; n < N; n++) {
                dI[c * N + n] = 0.0;
            }
            int rowstart = rowptr[c];
            int rowend = rowptr[c + 1];
            for (int i = rowstart; i < rowend; ++i) {
                int k =  colidx[i];
                float w = values[idxmap[i]];
                #pragma omp simd
                #pragma unroll
                for (int n = N_NB; n < N; n++) {
                    dI[c * N + n] += w * dO[k * N + n];    
                }
            }
        }
    }
}

// dO is a N x K matrix, stored in column major format
// I is a N x C matrix, stored in column major format
// dW is a C x K sparse matrix, stored in the CSC format
// dW = I_T x d_O
/*
   for (int k = 0; k < K; ++k) {
        int colstart = colptr[k];
        int colend = colptr[k + 1];
        for (int i = colstart; i < colend; ++i) {
            int c = rowidx[i];
            float dw = 0.0;
            for (int n = 0; n < N; ++n) {
                dw += I[c * N + n] * dO[k * N + n];
            }
            dvalues[i] = dw;
        }
    }
}
*/
template<int NB = 256>
void SpMLPUpd(int N,
              int K,
              int C,
              float *__restrict I,
              float *__restrict dO,
              int *__restrict colptr,
              int *__restrict rowidx,
              float *__restrict dvalues) { 
    int N_NB = N/NB * NB;
    #pragma omp parallel for
    for (int k = 0; k < K; ++k) {
        int colstart = colptr[k];
        int colend = colptr[k + 1];
        #pragma unroll
        for (int i = colstart; i < colend; ++i) {
            dvalues[i] = 0.0;
        }
    }
    for (int n = 0; n < N_NB; n += NB) {
        int nend = (n + NB > N ? N - n: NB);
        #pragma omp parallel for
        for (int k = 0; k < K; ++k) {
            int colstart = colptr[k];
            int colend = colptr[k + 1];
            for (int i = colstart; i < colend; ++i) {
                int c = rowidx[i];
                float dw = 0.0;
                #pragma omp simd
                #pragma unroll
                for (int nb = 0; nb < nend; ++nb) {
                    int nn = n + nb;
                    dw += I[c * N + nn] * dO[k * N + nn];
                }
                dvalues[i] += dw;
            }
        }
    }
    if (N_NB < N) {
        #pragma omp parallel for
        for (int k = 0; k < K; ++k) {
            int colstart = colptr[k];
            int colend = colptr[k + 1];
            for (int i = colstart; i < colend; ++i) {
                int c = rowidx[i];
                float dw = 0.0;
                #pragma omp simd
                #pragma unroll
                for (int n = N_NB; n < N; ++n) {
                    dw += I[c * N + n] * dO[k * N + n];
                }
                dvalues[i] += dw;
            }
        }
    }
}

void ComputeKIdx(int K,
                 int C,
                 int *__restrict colptr,
                 int *__restrict cidx,
                 int *__restrict kidx) {
    for (int k = 0; k < K; ++k) {
        int colstart = colptr[k];
        int colend = colptr[k + 1];
        for (int i = colstart; i < colend; ++i) {
            kidx[i] = k;
        }
    }
}

template<int NB = 160, int ZB = 4>
void SpMLPUpd2(int N,
               int nnz,
               float *__restrict I,
               float *__restrict dO,
               int *__restrict cidx,
               int *__restrict kidx,
               float *__restrict dvalues) {
    int N_NB = N/NB * NB;
    #pragma omp parallel for
    for (int i = 0; i < nnz; ++i) {
        dvalues[i] = 0.0;
    }
    int Z_ZB = nnz/ZB * ZB;
    for (int n = 0; n < N_NB; n += NB) {
        #pragma omp parallel for
        for (int i = 0; i < Z_ZB; i+=ZB) {
            int c_0 = cidx[i + 0];
            int k_0 = kidx[i + 0];
            int c_1 = cidx[i + 1];
            int k_1 = kidx[i + 1];
            int c_2 = cidx[i + 2];
            int k_2 = kidx[i + 2];
            int c_3 = cidx[i + 3];
            int k_3 = kidx[i + 3];
            __m512 v_dw_0 = _mm512_set1_ps(0.0f);
            __m512 v_dw_1 = _mm512_set1_ps(0.0f);
            __m512 v_dw_2 = _mm512_set1_ps(0.0f);
            __m512 v_dw_3 = _mm512_set1_ps(0.0f);
            for (int nb = 0; nb < NB; nb+=16) {
                int nn = n + nb;
                __m512 v_I_0 =   _mm512_loadu_ps(&I[c_0 * N + nn]);
                __m512 v_I_1 =   _mm512_loadu_ps(&I[c_1 * N + nn]);
                __m512 v_I_2 =   _mm512_loadu_ps(&I[c_2 * N + nn]);
                __m512 v_I_3 =   _mm512_loadu_ps(&I[c_3 * N + nn]);
                __m512 v_dO_0 = _mm512_loadu_ps(&dO[k_0 * N + nn]);
                __m512 v_dO_1 = _mm512_loadu_ps(&dO[k_1 * N + nn]);
                __m512 v_dO_2 = _mm512_loadu_ps(&dO[k_2 * N + nn]);
                __m512 v_dO_3 = _mm512_loadu_ps(&dO[k_3 * N + nn]);
                v_dw_0 = _mm512_fmadd_ps(v_I_0, v_dO_0, v_dw_0);
                v_dw_1 = _mm512_fmadd_ps(v_I_1, v_dO_1, v_dw_1);
                v_dw_2 = _mm512_fmadd_ps(v_I_2, v_dO_2, v_dw_2);
                v_dw_3 = _mm512_fmadd_ps(v_I_3, v_dO_3, v_dw_3);
            }
            dvalues[i + 0] += _mm512_reduce_add_ps(v_dw_0);
            dvalues[i + 1] += _mm512_reduce_add_ps(v_dw_1);
            dvalues[i + 2] += _mm512_reduce_add_ps(v_dw_2);
            dvalues[i + 3] += _mm512_reduce_add_ps(v_dw_3);
        }
    }
    if (Z_ZB < nnz) {
        for (int i = Z_ZB; i < nnz; ++i) {
            int c = cidx[i];
            int k = kidx[i];
            #pragma omp simd
            #pragma unroll
            for (int n = 0; n < N_NB; ++n) {
                dvalues[i] += I[c * N + n] * dO[k * N + n];
            }
        }    
    }
    if (N_NB < N) {
        #pragma omp parallel for
        for (int i = 0; i < nnz; ++i) {
            int c = cidx[i];
            int k = kidx[i];
            #pragma omp simd
            #pragma unroll
            for (int n = N_NB; n < N; ++n) {
                dvalues[i] += I[c * N + n] * dO[k * N + n];
            }
        }
    }
}

void BlockSpMatStep1(int K,
                     int C,
                     int KB,
                     int CB,
                     int* colptr,
                     int* rowidx,
                     unsigned int* b_colptr[],
                     int* nnzb) {
    assert(K%KB == 0);
    assert(C%CB == 0);
    int num_blocks = K/KB * C/CB;
    for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        nnzb[blk_idx] = 0;
        for (int i = 0; i <= KB; ++i) {
            b_colptr[blk_idx][i] = 0;
        }
    }
    for (int k = 0; k < K; ++k) {
        int k_blk_idx = k/KB;
        int k_blk_offset = k%KB;
        unsigned colstart = colptr[k];
        unsigned colend = colptr[k + 1];
        for (int i = colstart; i < colend; ++i) {
            int c = rowidx[i];
            int c_blk_idx = c/CB;
            int blk_idx = k_blk_idx * C/CB + c_blk_idx;
            nnzb[blk_idx]++;
            b_colptr[blk_idx][k_blk_offset + 1]++;
        }
    }
    for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        for (int i = 0; i < KB; ++i) {
            b_colptr[blk_idx][i + 1] += b_colptr[blk_idx][i];
        }
    }
}

void BlockSpMatStep2(int K,
                     int C,
                     int KB,
                     int CB,
                     int* colptr,
                     int* rowidx,
                     float* values,
                     unsigned int* b_colptr[],
                     unsigned int* b_rowidx[],
                     float* b_values[]) {
    assert(K%KB == 0);
    assert(C%CB == 0);
    int num_blocks = K/KB * C/CB;
    for (int k = 0; k < K; ++k) {
        int k_blk_idx = k/KB;
        int k_blk_offset = k%KB;
        unsigned colstart = colptr[k];
        unsigned colend = colptr[k + 1];
        for (int i = colstart; i < colend; ++i) {
            int c = rowidx[i];
            int c_blk_idx = c/CB;
            int c_blk_offset = c%CB;
            int blk_idx = k_blk_idx * C/CB+ c_blk_idx;
            b_rowidx[blk_idx][b_colptr[blk_idx][k_blk_offset]] = c_blk_offset;
            if (b_values != NULL) {
                b_values[blk_idx][b_colptr[blk_idx][k_blk_offset]] = values[i];
            }
            b_colptr[blk_idx][k_blk_offset]++;
        }
    }
    
    for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        for (int i = KB; i > 0; --i) {
            b_colptr[blk_idx][i] = b_colptr[blk_idx][i - 1];
        }
        b_colptr[blk_idx][0] = 0;
    }
}

#endif // KERNEL_H_