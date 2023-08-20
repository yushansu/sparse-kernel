#ifndef RMAT_H_
#define RMAT_H_

#include <cstdint>
#include "bloom.h"
#include "rnd_stream.h"

class RMat {
  public:
    RMat(unsigned int scale, unsigned efactor,
        double K_A = 0.59, double K_B = 0.19, double K_C = 0.19, bool scramble = true) {
        RT_CHECK(scale < 64);
        scale_ = scale;
        K_A_ = K_A;
        K_B_ = K_B;
        K_C_ = K_C;
        num_vertices_ = static_cast<unsigned long long>(1) << scale;
        RT_CHECK(efactor < num_vertices_);
        num_edges_ = num_vertices_ * efactor;
        // init random stream
        int nthreads = omp_get_max_threads();
        for (int i = 0; i < nthreads; ++i) {
            rnd_stream_.push_back(RndStream(i));
        }
        unsigned long long val0;
        unsigned long long val1;
        
        // init scramble list
        if (scramble) {
            scramble_list_ = new unsigned long long[num_vertices_];
            InitScrambleList(rnd_stream_[0], scramble_list_);
        }
        // init bloom filter
        double false_positive = 0.001;
        unsigned int num_funcs = static_cast<int>(log(1.0/false_positive)/log(2.0) + 1);
        size_t bf_size = (size_t)(num_edges_/log(2.0));
        bf_ = new BloomFilter(num_funcs, bf_size);
    }

    virtual ~RMat() {
        if (scramble_list_) {
            delete [] scramble_list_;
        }
        delete bf_;
    }

    void Generate(unsigned long long *src, unsigned long long *dst) {
        bf_->Reset();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            size_t chunk = (num_edges_ + nthreads - 1)/nthreads;
            chunk = (chunk % 2 == 0 ? chunk : chunk + 1);
            size_t start = chunk * tid;
            size_t end = (start + chunk > num_edges_ ? num_edges_ : chunk);
            size_t i = start;
            while (i < end) {
                unsigned long long ii = 0;
                unsigned long long jj = 0;
                Kronecker(rnd_stream_[tid], &ii, &jj);
                if (ii > jj) {
                    std::swap(ii, jj);
                }
                bool bf_flag = bf_->InsertKey(ii, jj);
                if (bf_flag == false && ii != jj) {
                    if (scramble_list_ != nullptr) {
                        src[i] = scramble_list_[ii];
                        dst[i] = scramble_list_[jj];
                        src[i + 1] = scramble_list_[jj];
                        dst[i + 1] = scramble_list_[ii];
                    } else {
                        src[i] = ii;
                        dst[i] = jj;
                        src[i + 1] = jj;
                        dst[i + 1] = ii;
                    }
                    i++;
                }
            }
        }
    }

  private:
    unsigned int scale_;
    double K_A_;
    double K_B_;
    double K_C_;
    unsigned long long num_vertices_;
    unsigned long long num_edges_;
    std::vector<RndStream> rnd_stream_;
    unsigned long long *scramble_list_ = nullptr;
    BloomFilter *bf_;
    
    void Kronecker(RndStream &stream, unsigned long long *src, unsigned long long *dst) {
        *src = 0;
        *dst = 0;
        for (int ib = scale_ - 1; ib >= 0; ib--) {
            unsigned long long ii;
            unsigned long long jj;
            double rand_num;
            stream.Uniformfp64(0.0, 1.0, 1, &rand_num);     
            if (rand_num < K_C_) {
                ii = 0;
                jj = 1;
            } else if (rand_num < K_B_ + K_C_) {
                ii = 1;
                jj = 0;
            } else if (rand_num < K_A_ + K_B_ + K_C_) {
                ii = 0;
                jj = 0;
            } else {
                ii = 1;
                jj = 1;
            }
            *src += (ii << ib);
            *dst += (jj << ib);
        } 
    }

    unsigned long long BitReverse(unsigned long long x) {
        uint32_t h = (uint32_t)(x >> 32);
        uint32_t l = (uint32_t)(x & UINT32_MAX);
        h = (h >> 16) | (h << 16);
        l = (l >> 16) | (l << 16);
        h = ((h >> 8) & UINT32_C(0x00FF00FF))|((h & UINT32_C(0x00FF00FF)) << 8);
        l = ((l >> 8) & UINT32_C(0x00FF00FF))|((l & UINT32_C(0x00FF00FF)) << 8);
        h = ((h >> 4) & UINT32_C(0x0F0F0F0F))|((h & UINT32_C(0x0F0F0F0F)) << 4);
        l = ((l >> 4) & UINT32_C(0x0F0F0F0F))|((l & UINT32_C(0x0F0F0F0F)) << 4);
        h = ((h >> 2) & UINT32_C(0x33333333))|((h & UINT32_C(0x33333333)) << 2);
        l = ((l >> 2) & UINT32_C(0x33333333))|((l & UINT32_C(0x33333333)) << 2);
        h = ((h >> 1) & UINT32_C(0x55555555))|((h & UINT32_C(0x55555555)) << 1);
        l = ((l >> 1) & UINT32_C(0x55555555))|((l & UINT32_C(0x55555555)) << 1);
	    return ((unsigned long long)l << 32) | h;
    }

    void InitScrambleList(RndStream &stream, unsigned long long *scramble_list_) {      
        unsigned long long val0;
        unsigned long long val1; 
        stream.Uniformint64(1, &val0);
        stream.Uniformint64(1, &val1);
        #pragma omp parallel for
        for (unsigned long long i = 0; i < num_vertices_; i++) {
            unsigned long long v_new = i + val0 + val1;
            v_new *= val0 | UINT64_C(0x4519840211493211);
            v_new = (BitReverse(v_new) >> (64 - scale_));
            RT_CHECK((v_new >> scale_) == 0);
            v_new *= val1 | UINT64_C(0x3050852102C843A5);
            v_new = (BitReverse(v_new) >> (64 - scale_));
            RT_CHECK((v_new >> scale_) == 0);
            scramble_list_[i] = v_new;
        }
    }
};

#endif // RMAT_H_