#ifndef BLOCKCOO_H_
#define BLOCKCOO_H_

#include <string>
#include <fstream>
#include <cstdlib>
#include <omp.h>
#include "utils.h"
#include "memory.h"

template <typename IdxType, typename ValType>
class BlockCOO {
    static_assert(std::is_integral<IdxType>::value, "Not supported");
    static_assert(std::is_floating_point<ValType>::value || std::is_integral<ValType>::value,
                  "Not supported");
  public:
    BlockCOO(size_t brows, size_t bcols, size_t nnzb, size_t bsize)
        : brows_(brows), bcols_(bcols), nnzb_(nnzb), bsize_(bsize) {
        Allocate();
    }
    
    BlockCOO(size_t brows, size_t bcols, size_t nnzb,
             IdxType *browidx, IdxType *bcolidx, IdxType *bvalues)
        : wrapped_(true), brows_(brows), bcols_(bcols), nnzb_(nnzb),
          browidx_(browidx), bcolidx_(bcolidx), bvalues_(bvalues) {
    }
    
    explicit BlockCOO(const std::string &mtx_file, size_t bsize) : bsize_(bsize) {
        std::ifstream fin;
        fin.open(mtx_file.c_str());
        // read banner
         // read banner
        MM_typecode matcode;
        size_t lines = 0;
        int ret = MTXUtils::MMReadHeader(fin, &matcode, &brows_, &bcols_, &lines);
        RT_CHECK(ret == 0, ret);
        RT_CHECK(mm_is_sparse(matcode));
        RT_CHECK(mm_is_pattern(matcode));
        nnzb_ = (mm_is_symmetric(matcode) ? 2 * lines : lines);
        // allocate memory
        Allocate();
        // read nonzeros
        ret = MTXUtils::MMReadMtxCrd(fin, matcode, brows_, bcols_, lines,
                                            &nnzb_, browidx_, bcolidx_,  (ValType *)(nullptr),
                                            mm_is_symmetric(matcode));
        for (size_t i = 0; i < nnzb_ * bsize_ * bsize_; i++) {
            bvalues_[i] = static_cast<ValType>(10.0 * rand()/RAND_MAX);
        }
        fin.close();
    }

    // copy constructor
    BlockCOO(const BlockCOO<IdxType, ValType> &other)
        : brows_(other.brows_), bcols_(other.bcols_), nnzb_(other.nnzb_), bsize_(other.bsize_) {
        Allocate();
        std::copy_n(other.browidx_, browidx_, nnzb_);
        std::copy_n(other.bcolidx_, bcolidx_, nnzb_);
        std::copy_n(other.bvalues_, bvalues_, nnzb_ * bsize_);
    }

    // move constructor
    BlockCOO(const BlockCOO<IdxType, ValType> &&other)
        : brows_(other.brows_), bcols_(other.bcols_), nnzb_(other.nnzb_), bsize_(other.bsize_) {
        browidx_ = other.browidx_;
        bcolidx_ = other.bcolidx_;
        bvalues_ = other.bvalues_;
        other->brows_ = 0;
        other->bcols_ = 0;
        other->bnnz_ = 0;
        other->bsize_ = 0;
        other.browidx_ = nullptr;
        other.bcolidx_ = nullptr;
        other.bvalues_ = nullptr;
    }

    // assignment operator
    BlockCOO<IdxType, ValType> &operator=(const BlockCOO<IdxType, ValType> &other) {
        if (this != &other) {
            brows_ = other.brows_;
            bcols_ = other.bcols_;
            nnzb_ = other.nnzb_;
            bsize_ = other.bsize_;
            Destroy();
            if (other.browidx_ && other.bcolidx_ && other.bvalues_) {
                Allocate();
                std::copy_n(other.browidx_, browidx_, nnzb_);
                std::copy_n(other.bcolidx_, bcolidx_, nnzb_);
                std::copy_n(other.bvalues_, bvalues_, nnzb_); 
            } else {
                browidx_ = nullptr;
                bcolidx_ = nullptr;
                bvalues_ = nullptr;
            }            
        }
        return *this;
    }

    // move assignment operator
    BlockCOO<IdxType, ValType> &operator=(const BlockCOO<IdxType, ValType> &&other) {
        if (this != &other) {
            brows_ = other.brows_;
            bcols_ = other.bcols_;
            nnzb_ = other.nnzb_;
            bsize_ = other.bsize_;
            Destroy();
            browidx_ = other.browidx_;
            bcolidx_ = other.bcolidx_;
            bvalues_ = other.bvalues_;
            other.brows_ = 0;
            other.bcols_ = 0;
            other.nnzb_ = 0;
            other->bsize_ = 0;
            other.browidx_ = nullptr;
            other.bcolidx_ = nullptr;
            other.bvalues_ = nullptr;
        }
        return *this;
    }

    ~BlockCOO() {
        Destroy();
    }

    size_t brows() const {
        return brows_;
    }

    size_t bcols() const {
        return bcols_;
    }

    size_t nnzb() const {
        return nnzb_;
    }

    size_t bsize() const {
        return bsize_;
    }

    IdxType* browidx() const {
        return browidx_;
    }

    IdxType* bcolidx() const {
        return bcolidx_;
    }

    ValType* bvalues() const {
        return bvalues_;
    }
    
    void StoreMatrixMarket(const std::string &mtx_file, bool index_only=true) const {
        // open mtx file
        std::ofstream fout;
        fout.open(mtx_file.c_str());
        if (index_only) {
            MTXUtils::MMWriteHeader(fout, brows_, bcols_, nnzb_, true);
            // write values
            for (size_t i = 0; i < nnzb_; ++i) {
                fout << browidx_[i] + 1 << " " << bcolidx_[i] + 1 << endl;
            }
        } else {
            MTXUtils::MMWriteHeader(fout, brows_ * bsize_, bcols_ * bsize_,
                                           nnzb_ * bsize_ * bsize_);
            for (size_t i = 0; i < nnzb_; ++i) {
                for (size_t bx = 0; bx < bsize_; ++bx) {
                    for (size_t by = 0; by < bsize_; ++by) {
                        size_t x = browidx_[i] * bsize_ + bx;
                        size_t y = bcolidx_[i] * bsize_ + by;
                        ValType val = bvalues_[i * bsize_ * bsize_ + bx * bsize_ + by];
                        fout << x << " " << y << " " << val <<std::endl;
                    }
                }
            } // for (size_t i = 0; i < nnzb_; ++i)
        }
        fout.close();
    }

    //! Standard output
    friend std::ostream &operator<<(std::ostream &os, const BlockCOO<IdxType, ValType> &bcoo) {
        if (bcoo.nnzb_ == 0) {
            os << "Empty matrix" <<std::endl;
            return os;
        }
        if (bcoo.wrapped_) {
            os << "Wrapped" <<std::endl;
        }
        os << bcoo.brows_ * bcoo.bsize_ << " " << bcoo.bcols_ * bcoo.bsize_ << " "
           << bcoo.nnzb_ * bcoo.bsize_ * bcoo.bsize_ << " " <<std::endl;
        for (size_t i = 0; i < bcoo.nnzb_; ++i) {
            for (size_t bx = 0; bx < bcoo.bsize_; ++bx) {
                for (size_t by = 0; by < bcoo.bsize_; ++by) {
                    size_t x = bcoo.browidx_[i] * bcoo.bsize_ + bx;
                    size_t y = bcoo.bcolidx_[i] * bcoo.bsize_ + by;
                    ValType val = bcoo.bvalues_[i * bcoo.bsize_ * bcoo.bsize_ + bx * bcoo.bsize_ + by];
                    os << "(" << x << ", " << y << "):" << val <<std::endl;
                }
            }
        }
        return os;
    }

  private:
    bool wrapped_ = false;
    size_t brows_;
    size_t bcols_;
    size_t bsize_;
    size_t nnzb_;
    IdxType *browidx_ = nullptr;
    IdxType *bcolidx_ = nullptr;
    ValType *bvalues_ = nullptr;

    void Allocate() {
        RT_CHECK(nnzb_ > 0);
        RT_CHECK(bsize_ > 0);
        browidx_ = MMUtils::Alloc<IdxType>(nnzb_);
        bcolidx_ = MMUtils::Alloc<IdxType>(nnzb_);
        bvalues_ = MMUtils::Alloc<ValType>(nnzb_ * bsize_ * bsize_);
        RT_CHECK(browidx_);
        RT_CHECK(bcolidx_);
        RT_CHECK(bvalues_);
    }

    void Destroy() {
        if (!wrapped_) {
            MMUtils::Free(browidx_);
            MMUtils::Free(bcolidx_);
            MMUtils::Free(bvalues_);
        }
    }
};

#endif // COO_H_