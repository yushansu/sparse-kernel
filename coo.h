#ifndef COO_H_
#define COO_H_

#include <string>
#include <fstream>
#include <cstdlib>
#include <omp.h>
#include "mmio.h"
#include "exception.h"
#include "memory.h"

template <typename IdxType, typename ValType>
class COO {
    static_assert(std::is_integral<IdxType>::value, "Not supported");
    static_assert(std::is_floating_point<ValType>::value || std::is_integral<ValType>::value,
                  "Not supported");
  public:
    COO(size_t rows, size_t cols, size_t nnz) : rows_(rows), cols_(cols), nnz_(nnz) {
        Allocate();
    }

    COO(size_t rows, size_t cols, size_t nnz,
        IdxType rowidx, IdxType *colidx, IdxType *values)
        : wrapped_(true), rows_(rows), cols_(cols), nnz_(nnz),
          rowidx_(rowidx), colidx_(colidx), values_(values) {
    }

    explicit COO(const std::string &mtx_file, bool force_symmetric=false) {
        std::ifstream fin;
        fin.open(mtx_file.c_str(), std::ifstream::in);
        RT_CHECK(fin.is_open());
        // read banner
        MM_typecode matcode;
        size_t lines = 0;
        MTXUtils::Status ret = MTXUtils::MMReadHeader(fin, &matcode, &rows_, &cols_, &lines);
        RT_CHECK(ret == MTXUtils::Status::MM_OK, ret);
        RT_CHECK(mm_is_sparse(matcode));
        bool is_symmetric = force_symmetric || mm_is_symmetric(matcode);
        nnz_ = (is_symmetric ? 2 * lines : lines);
        // allocate memory
        Allocate();
        // read nonzeros
        ret = MTXUtils::MMReadMtxCrd(fin, matcode, rows_, cols_, lines,
                                     &nnz_, rowidx_, colidx_, values_,
                                     is_symmetric);
        RT_CHECK(ret == MTXUtils::Status::MM_OK, ret);
        if (mm_is_pattern(matcode)) {
            for (size_t i = 0; i < nnz_; i++) {
                values_[i] = static_cast<ValType>(10.0 * rand()/RAND_MAX);
            }
        }
    }

    // copy constructor
    COO(const COO<IdxType, ValType> &other)
        : rows_(other.rows_), cols_(other.cols_), nnz_(other.nnz_) {
        Allocate();
        std::copy_n(other.rowidx_, rowidx_, nnz_);
        std::copy_n(other.colidx_, colidx_, nnz_);
        std::copy_n(other.values_, values_, nnz_);
    }

    // move constructor
    COO(const COO<IdxType, ValType> &&other)
        : rows_(other.rows_), cols_(other.cols_), nnz_(other.nnz_) {
        rowidx_ = other.rowidx_;
        colidx_ = other.colidx_;
        values_ = other.values_;
        other->rows_ = 0;
        other->cols_ = 0;
        other->nnz_ = 0;
        other.rowidx_ = nullptr;
        other.colidx_ = nullptr;
        other.values_ = nullptr;
    }

    // assignment operator
    COO<IdxType, ValType> &operator=(const COO<IdxType, ValType> &other) {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            nnz_ = other.nnz_;
            Destroy();
            if (other.rowidx_ && other.colidx_ && other.values_) {
                Allocate();
                std::copy_n(other.rowidx_, rowidx_, nnz_);
                std::copy_n(other.colidx_, colidx_, nnz_);
                std::copy_n(other.values_, values_, nnz_); 
            } else {
                rowidx_ = nullptr;
                colidx_ = nullptr;
                values_ = nullptr;
            }            
        }
        return *this;
    }

    // move assignment operator
    COO<IdxType, ValType> &operator=(const COO<IdxType, ValType> &&other) {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            nnz_ = other.nnz_;
            Destroy();
            rowidx_ = other.rowidx_;
            colidx_ = other.colidx_;
            values_ = other.values_;
            other.rows_ = 0;
            other.cols_ = 0;
            other.nnz_ = 0;
            other.rowidx_ = nullptr;
            other.colidx_ = nullptr;
            other.values_ = nullptr;
        }
        return *this;
    }

    virtual ~COO() { 
        Destroy();
    }
    
    size_t rows() const {
        return rows_;
    }

    size_t cols() const {
        return cols_;
    }

    size_t nnz() const {
        return nnz_;
    }

    IdxType* rowidx() const {
        return rowidx_;
    }

    IdxType* colidx() const {
        return colidx_;
    }

    ValType* values() const {
        return values_;
    }

    void StoreMatrixMarket(const std::string &mtx_file) const {
        // open mtx file
        std::ofstream fout;
        fout.open(mtx_file.c_str());
        MTXUtils::MMWriteHeader(fout, rows_, cols_, nnz_, true);
        // print values
        for (size_t i = 0; i < nnz_; ++i) {
            fout << rowidx_[i] + 1 << " " << colidx_[i] + 1 << " " << values_[i] <<std::endl;
        }
        fout.close();
    }

    void ToCSR(IdxType *rowptr, IdxType *colidx, ValType *values) const {
        // init rowptr
        for (size_t i = 0; i <= rows_; ++i) {
            rowptr[i] = 0;
        }
        // determine row lengths
        for (size_t i = 0; i < nnz_; ++i) {
            rowptr[rowidx_[i] + 1]++;
        }
        for (size_t i = 0; i < rows_; ++i) {
            rowptr[i + 1] += rowptr[i];
        }
        // go through the structure  once more. Fill in output matrix
        for (size_t k = 0; k < nnz_; k++) {
            values[ rowptr[rowidx_[k]] ] = values_[k];
            colidx[ rowptr[rowidx_[k]] ] = colidx_[k];
            rowptr[rowidx_[k]]++;
        }
        // shift back rowptr
        for (size_t i = rows_; i > 0; --i) {
            rowptr[i] = rowptr[i - 1];
        }
        rowptr[0] = 0;
        // sort colidx
        #pragma omp parallel for
        for (size_t i = 0; i < rows_; i++) {
            detail::SortDict<IdxType, ValType>(colidx, values, rowptr[i], rowptr[i + 1]);
        }
    }

    //! Standard output
    friend std::ostream &operator<<(std::ostream &os, const COO<IdxType, ValType> &coo) {
        if (coo.nnz_ == 0) {
            os << "<Empty matrix>" <<std::endl;
            return os;
        }
        if (coo.wrapped_) {
            os << "<Wrapped>" <<std::endl;
        }

        os << coo.rows_ << "x" << coo.cols_ << " " << coo.nnz_ << " " <<std::endl;
        for (size_t i = 0; i < coo.nnz_; ++i) {
            os << "(" << coo.rowidx_[i] << ", " << coo.colidx_[i] << "): " << coo.values_[i] <<std::endl;
        }
        return os;
    }

  private:
    bool wrapped_ = false;
    size_t rows_ = 0;
    size_t cols_ = 0;
    size_t nnz_ = 0;
    IdxType* rowidx_ = nullptr;
    IdxType* colidx_ = nullptr;
    ValType* values_ = nullptr;
    
    //! Helper function used by constructor
    void Allocate() {
        RT_CHECK(nnz_ > 0);
        rowidx_ = MMUtils::Alloc<IdxType>(nnz_);
        colidx_ = MMUtils::Alloc<IdxType>(nnz_);
        values_ = MMUtils::Alloc<ValType>(nnz_);
        RT_CHECK(rowidx_);
        RT_CHECK(colidx_);
        RT_CHECK(values_);
    }

    void Destroy() {
        if (!wrapped_) {
            MMUtils::Free(rowidx_);
            MMUtils::Free(colidx_);
            MMUtils::Free(values_);
        }
    }
}; // class COO

#endif // COO_H_