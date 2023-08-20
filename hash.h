#ifndef HASH_H_
#define HASH_H_

#include <cstdint>
#include "memory.h"

template<typename T, typename B, typename O>
class HashMap {
  static_assert(std::is_unsigned<T>::value, "Only defined for unsigned integral types");

  public:
    HashMap(size_t max_size) {
        max_size_ = max_size;
        key_array_ = MMUtils::Alloc<T>(max_size);
        base_array_ = MMUtils::Alloc<B>(max_size);
        offset_array_ = MMUtils::Alloc<O>(max_size);
        assert(key_array_);
        assert(base_array_);
        assert(offset_array_);
        Clear();
    }

    virtual ~HashMap() {
        MMUtils::Free(key_array_);
        MMUtils::Free(base_array_);
        MMUtils::Free(offset_array_);
    }

    size_t max_size() const {
        return max_size_;
    }

    size_t end() const {
        return max_size_;
    }

    size_t size() {
        return size_;
    }

    void Clear() {
        #pragma omp parallel for
        for (size_t i = 0; i < max_size_; ++i) {
            key_array_[i] = EMPTY;
        }
        size_ = 0;
    }

    size_t FindAndInsert(T key, B base, O offset) {
        T start = IntHash(key);
        for (size_t i = start; i < start + max_size_; ++i) {
            size_t pos = i % max_size_;
            if (key_array_[pos] == EMPTY) {
                T prev_key = __sync_val_compare_and_swap(&key_array_[pos], EMPTY, key);
                if (prev_key == EMPTY) {
                    base_array_[pos] = base;
                    offset_array_[pos] = offset;
                    return max_size_;
                } else if (prev_key == key) {
                    return pos;
                }
            } else if (key_array_[pos] == key) {
                return pos;
            }
            start++;
        }
        return max_size_;
    }

    size_t Query(T key, B *base, O *offset) {
        T start = IntHash(key);
        for (size_t i = start; i < start + max_size_; ++i) {
            size_t pos = i % max_size_;
            if (key_array_[pos] == EMPTY) {
                return max_size_;
            } else if (key_array_[pos] == key) {
                *base = base_array_[pos];
                *offset = offset_array_[pos];
                return pos;
            }
            start++;
        }
        return max_size_;
    }

  private:
    enum {
      EMPTY = ~T(0) 
    };

    T *key_array_ = nullptr;
    B *base_array_ = nullptr;
    O *offset_array_ = nullptr;
    size_t max_size_ = 0;
    size_t size_ = 0;

    uint32_t IntHash(uint32_t k) {
        k = ((k >> 16) ^ k) * 0x45d9f3b;
        k = ((k >> 16) ^ k) * 0x45d9f3b;
        k = (k >> 16) ^ k;
        return k;
    }

    uint64_t IntHash(uint64_t k) {
        k = (k ^ (k >> 30)) * 0xbf58476d1ce4e5b9;
        k = (k ^ (k >> 27)) * 0x94d049bb133111eb;
        k = k ^ (k >> 31);
        return k;
    }
};

#endif // HASH_H_