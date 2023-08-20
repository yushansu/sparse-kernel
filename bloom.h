#ifndef BLOOM_H_
#define BLOOM_H_

#include <mutex>
#include <cstdio>
#include "exception.h"

class BloomFilter {

  public:
    BloomFilter(unsigned int num_funcs, size_t bf_size) {
        bf_size_ = bf_size;
        size_in_bytes_ = (bf_size_ + 7)/8;
        nfuncs_ = num_funcs;
        bf_array_ = MMUtils::Alloc<unsigned char, nfuncs_ * size_in_bytes_);
        RT_CHECK(bf_array_);
        Reset();
    }

    virtual ~BloomFilter() {
        MMUtils::Free(bf_array_);
    }

    void Reset() {
        memset(bf_array_, 0, size_in_bytes_ * nfuncs_ * sizeof(unsigned char));
    }

    bool InsertKey(unsigned long long src, unsigned long long dst) {
        unsigned long long hash_value[nfuncs_];
        for (unsigned int i = 0; i < nfuncs_; i++) {
            unsigned long long hash_value0 = MurmurHash(src, dst, bf_size_, 0);
            unsigned long long hash_value1 = MurmurHash(src, dst, bf_size_, hash_value0);
            hash_value[i] = (hash_value0 + i * hash_value1) % bf_size_;
        }
        unsigned char (* bf_array)[size_in_bytes_] = (unsigned char (*)[*])bf_array_;
        bool exsit = true;
        for (unsigned int i = 0; i < nfuncs_; i++) {
            unsigned char val = static_cast<unsigned char>(1) << (hash_value[i] % 8);
            unsigned char bit = __sync_fetch_and_or(bf_array[i] + hash_value[i]/8, val);
            bit &= val;
            if (bit == 0) {
                exsit = false;
            }
        }
        return exsit;
    }

  private:
    unsigned char *bf_array_;
    size_t bf_size_;
    size_t size_in_bytes_;
    unsigned int nfuncs_;
    std::mutex array_mutex_;
    enum {
      HASH_M = 0xc6a4a7935bd1e995,
      HASH_R = 47
    };
    
    unsigned long long MurmurHash(unsigned long long src, unsigned long long dst,
                        unsigned long long len, unsigned long long seed) {
        unsigned long long h = seed ^ (16 * HASH_M);
        unsigned long long k0 = src;
        unsigned long long k1 = dst;
        k0 *= HASH_M;
        k0 ^= k0 >> HASH_R;
        k0 *= HASH_M;
        k1 *= HASH_M;
        k1 ^= k1 >> HASH_R;
        k1 *= HASH_M;
        h ^= k0;
        h *= HASH_M;
        h ^= k1;
        h *= HASH_M;
        h ^= h >> HASH_R;
        h *= HASH_M;
        h ^= h >> HASH_R;
        return (h % len);
    }
};

#endif // BLOOM_H_