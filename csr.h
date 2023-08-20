#ifndef CSR_H_
#define CSR_H_

#include <string>
#include <omp.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#ifdef HAVE_MKL_
#include <mkl_spblas.h>
#include <mkl.h>
#endif
#include "coo.h"
#include "memory.h"

template <typename IdxType, typename ValType>
class CSR {
    static_assert(std::is_integral<IdxType>::value, "Not supported");
    static_assert(std::is_floating_point<ValType>::value || std::is_integral<ValType>::value,
                  "Not supported");

  public:
    CSR() {}
    
    CSR(size_t rows, size_t cols, size_t nnz)
         : rows_(rows), cols_(cols), nnz_(nnz) {
        Allocate();
    }
    
    CSR(size_t rows, size_t cols, size_t nnz,
        IdxType rowptr, IdxType *colidx, IdxType *values)
        : wrapped_(true), rows_(rows), cols_(cols), nnz_(nnz),
          rowptr_(rowptr), colidx_(colidx), values_(values) {
        dptr_ = MMUtils::Alloc<IdxType>(rows_ + 1); 
        didx_ = MMUtils::Alloc<IdxType>(rows_);
        RT_CHECK(dptr_);
        RT_CHECK(didx_);
        InitRowTable();
    }

    explicit CSR(const std::string &mtx_file, bool force_symmetric = false) {
        if(mtx_file.substr(mtx_file.find_last_of(".") + 1) != "bin") {
            COO<IdxType, ValType> coo(mtx_file, force_symmetric);
            rows_ = coo.rows();
            cols_ = coo.cols();
            nnz_ = coo.nnz();
            Allocate();
            coo.ToCSR(rowptr_, colidx_, values_);
            InitRowTable();
        } else {
            std::ifstream fin;
            fin.open(mtx_file.c_str(), std::ifstream::in);
            RT_CHECK(fin.is_open());
            fin.read(reinterpret_cast<char *>(&rows_), sizeof(size_t));
            fin.read(reinterpret_cast<char *>(&cols_), sizeof(size_t));
            fin.read(reinterpret_cast<char *>(&nnz_), sizeof(size_t));
            Allocate();
            fin.read(reinterpret_cast<char *>(rowptr_), (rows_ + 1) * sizeof(IdxType));
            fin.read(reinterpret_cast<char *>(colidx_), nnz_ * sizeof(IdxType));
            fin.close();
            InitRowTable();
        }  
    }
    
    // copy constructor
    CSR(const CSR<IdxType, ValType> &other)
        : rows_(other.rows_), cols_(other.cols_), nnz_(other.nnz_), nnzr_(other.nnzr_), wrapped_(false) {
        if (!other.wrapped_) {
            if (other.nnz_ > 0) {
                Allocate();
                std::copy_n(other.rowptr_, rows_ + 1, rowptr_);
                std::copy_n(other.colidx_, nnz_, colidx_);
                std::copy_n(other.values_, nnz_, values_);
                std::copy_n(other.dptr_, nnzr_, dptr_);
                std::copy_n(other.didx_, nnzr_, didx_);
            } // if (other.nnz_ > 0)
        } else {
            rowptr_ = other.rowptr_;
            colidx_ = other.rowptr_;
            values_ = other.rowptr_;
            dptr_ = other.dptr_;
            didx_ = other.didx_;
        } // if (!other.wrapped_)
    }

    // move constructor
    CSR(const CSR<IdxType, ValType> &&other)
        : rows_(other.rows_), cols_(other.cols_), nnz_(other.nnz_), nnzr_(other.nnzr_), wrapped_(other.wrapped_),
          rowptr_(other.rowptr_), colidx_(other.colidx_), values_(other.values_),
          dptr_(other.dptr_), didx_(other.didx_) {
        other.rows_ = 0;
        other.cols_ = 0;
        other.nnz_ = 0;
        other.nnzr_ = 0;
        other.wrapped_ = false; 
        other.rowptr_ = nullptr;
        other.colidx_ = nullptr;
        other.values_ = nullptr;
        other.dptr_ = nullptr;
        other.didx_ = nullptr;
    }

    // assignment operator
    CSR<IdxType, ValType> &operator=(const CSR<IdxType, ValType> &other) {
        if (this != &other) {
            Destroy();
            rows_ = other.rows_;
            cols_ = other.cols_;
            nnz_ = other.nnz_;
            nnzr_ = other.nnzr_;
            wrapped_ = other.wrapped_;
            if (!other.wrapped_) { 
                if (other.nnz_ > 0) {
                    Allocate();
                    std::copy_n(other.rowptr_, rows_ + 1, rowptr_);
                    std::copy_n(other.colidx_, nnz_, colidx_);
                    std::copy_n(other.values_, nnz_, values_);
                    std::copy_n(other.dptr_, dptr_, nnzr_);
                    std::copy_n(other.didx_, didx_, nnzr_);
                } // if (other.nnz_ > 0)
            } else {
                rowptr_ = other.rowptr_;
                colidx_ = other.colidx_;
                values_ = other.values_;
                dptr_ = other.dptr_;
                didx_ = other.didx_;
            } // if (!other.wrapped_) 
        } // if (this != &other)
        return *this;
    }

    // move assignment operator
    CSR<IdxType, ValType> &operator=(const CSR<IdxType, ValType> &&other) {
        if (this != &other) {
            Destroy();
            rows_ = other.rows_;
            cols_ = other.cols_;
            nnz_ = other.nnz_;
            nnzr_ = other.nnzr_;
            wrapped_ = other.wrapped_;
            rowptr_ = other.rowptr_;
            colidx_ = other.colidx_;
            values_ = other.values_;
            dptr_ = other.dptr_;
            didx_ = other.didx_;
            // reset other to default
            other.rows_ = 0;
            other.cols_ = 0;
            other.nnz_ = 0;
            other.nnzr_ = 0;
            other.wrapped_ = false;
            other.rowptr_ = nullptr;
            other.colidx_ = nullptr;
            other.values_ = nullptr;
            other.dptr_ = nullptr;
            other.didx_ = nullptr;
        }
        return *this;
    }

    virtual ~CSR() {
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

    size_t nnzr() const {
        return nnzr_;
    }

    IdxType* rowptr() const {
        return rowptr_;
    }
    
    IdxType rowptr(size_t i) const {
        return rowptr_[i];
    }

    IdxType* colidx() const {
        return colidx_;
    }

    IdxType colidx(size_t i) const {
        return colidx_[i];
    }

    ValType* values() const {
        return values_;
    }

    ValType values(size_t i) const {
        return values_[i];
    }

    IdxType *dptr() const {
        return dptr_;
    }
    
    IdxType *dptr(size_t i) const {
        return dptr_[i];
    }

    IdxType *didx() const {
        return didx_;
    }

    IdxType *didx(size_t i) const {
        return didx_[i];
    }

#ifdef HAVE_MKL_
    sparse_matrix_t mkl_matrix() const {
        return mkl_matrix_;
    }

    struct matrix_descr mkl_descr() const {
        return mkl_descr_;
    }
#endif

    void StoreMatrixMarket(const std::string &mtx_file, bool compressed = false) const {
        // open mtx file
        std::ofstream fout;
        fout.open(mtx_file.c_str(), std::ifstream::out);
        RT_CHECK(fout.is_open());
        if (compressed) {
            MTXUtils::MMWriteHeader(fout, nnzr_, cols_, nnz_, true);
            // print values
            size_t rowstart = dptr_[0];
            for (size_t i = 0; i < nnzr_; ++i) {
                size_t rowend = dptr_[i + 1];
                for (size_t j = rowstart; j < rowend; j++) {
                    fout << i + 1 << " " << colidx_[j] + 1 << " " << values_[j] << std::endl;
                }
                rowstart = rowend;
            }
            std::ofstream faux;
            faux.open( (mtx_file + ".aux").c_str() );
            faux << nnzr_ << std::endl;
            for (size_t i = 0; i < nnzr_; ++i) {
                faux << didx_[i] + 1 << std::endl;
            }
            faux.close();
        } else {
            MTXUtils::MMWriteHeader(fout, rows_, cols_, nnz_, true);
            // print values
            size_t rowstart = rowptr_[0];
            for (size_t i = 0; i < rows_; ++i) {
                size_t rowend = rowptr_[i + 1];
                for (size_t j = rowstart; j < rowend; j++) {
                    fout << i + 1 << " " << colidx_[j] + 1 << " " << values_[j] << std::endl;
                }
                rowstart = rowend;
            }
        }   
        fout.close();
    }

    void StoreBinary(const std::string &bin_file) const {
        std::ofstream fout;
        fout.open(bin_file.c_str(), std::ifstream::out | std::ifstream::trunc);
        RT_CHECK(fout.is_open());
        fout.write(reinterpret_cast<const char *>(&rows_), sizeof(size_t));
        fout.write(reinterpret_cast<const char *>(&cols_), sizeof(size_t));
        fout.write(reinterpret_cast<const char *>(&nnz_), sizeof(size_t));
        fout.write(reinterpret_cast<const char *>(rowptr_), (rows_ + 1) * sizeof(IdxType));
        fout.write(reinterpret_cast<const char *>(colidx_), nnz_ * sizeof(IdxType));
        fout.close();        
    }
    
    void InitRowTable() {
        nnzr_ = 0;
        for (size_t i = 0; i < rows_; ++i) {
            if (rowptr_[i + 1] != rowptr_[i]) {
                dptr_[nnzr_] =  rowptr_[i];
                didx_[nnzr_] = i;
                nnzr_++;
            }
        }
        dptr_[nnzr_] = nnz_;
    }

    // Transposition using parallel counting sort
    CSR<IdxType, ValType> *Transpose() const {
        // construct A_T
        CSR<IdxType, ValType>  *AT = new CSR<IdxType, ValType>(cols_, rows_, nnz_);
        IdxType *rowptr_T = AT->rowptr();
        IdxType *colidx_T = AT->colidx();
        ValType *values_T = AT->values();
        // init rowptr
        for (size_t i = 0; i <= cols_; ++i) {
            rowptr_T[i] = 0;
        }
        // determine row lengths
        for (size_t i = 0; i < nnz_; ++i) {
            rowptr_T[colidx_[i] + 1]++;
        }
        for (size_t i = 0; i < cols_; ++i) {
            rowptr_T[i + 1] += rowptr_T[i];
        }
        // go through the structure  once more. Fill in output matrix
        size_t rowstart = 0;
        for (size_t i = 0; i < rows_; ++i) {
            size_t rowend = rowptr_[i + 1];
            for (size_t j = rowstart; j < rowend; j++) {
                values_T[ rowptr_T[colidx_[j]] ] = values_[j];
                colidx_T[ rowptr_T[colidx_[j]] ] = i;
                rowptr_T[colidx_[j]]++;
            }
            rowstart = rowend;
        }
        // shift back rowptr
        for (size_t i = cols_; i > 0; --i) {
            rowptr_T[i] = rowptr_T[i - 1];
        }
        rowptr_T[0] = 0;
        // sort colidx
        for (size_t i = 0; i < cols_; i++) {
            detail::SortDict<IdxType, ValType>(colidx_T, values_T, rowptr_T[i], rowptr_T[i + 1]);
        }
        AT->InitRowTable();
        return AT;
    }

    // Transposition using parallel counting sort
    CSR<IdxType, ValType> *Slice(size_t start_row, size_t num_rows) const {
        RT_CHECK(num_rows >= 0);
        RT_CHECK(start_row >= 0);
        RT_CHECK(start_row + num_rows < rows_);
        size_t nnz = rowptr_[start_row + num_rows] - rowptr_[start_row];
        // construct A_S
        CSR<IdxType, ValType>  *AS = new CSR<IdxType, ValType>(num_rows, cols_, nnz);
        IdxType *rowptr_S = AS->rowptr();
        IdxType *colidx_S = AS->colidx();
        ValType *values_S = AS->values();
        // init rowptr
        size_t base = rowptr_[start_row];
        for (size_t i = 0; i <= num_rows; ++i) {
            rowptr_S[i] = rowptr_[start_row + i] - base;
        }
        RT_CHECK(rowptr_S[num_rows] == nnz);
        for (size_t i = 0; i < nnz; ++i) {
            colidx_S[i] = colidx_[i + base];
            values_S[i] = values_[i + base];
        }
        AS->InitRowTable();
        return AS;
    }

    //! Standard output
    friend std::ostream &operator<<(std::ostream &os, const CSR<IdxType, ValType> &csr) {
        if (csr.nnz_ == 0) {
            os << "<Empty matrix>" << std::endl;
            return os;
        }
        if (csr.wrapped_) {
            os << "<Wrapped>" << std::endl;
        }
        os << csr.rows_ << "x" << csr.cols_ << " " << csr.nnz_ << " " << std::endl;
        size_t rowstart = csr.rowptr_[0];
        for (size_t i = 0; i < csr.rows_; ++i) {
            size_t rowend = csr.rowptr_[i + 1];
            for (size_t j = rowstart; j < rowend; j++) {
                os << "(" << i << ", " << csr.colidx_[j] << "): " << csr.values_[j] << std::endl;
            }
            rowstart = rowend;
        }
        return os;
    }

  private:
    bool hugepage_ = false;
    bool hugepage_1g_ = false;
    bool wrapped_ = false;
    size_t rows_ = 0;
    size_t cols_  = 0;
    size_t nnz_ = 0;
    size_t nnzr_ = 0;
    IdxType* rowptr_ = nullptr;
    IdxType* colidx_ = nullptr;
    ValType* values_ = nullptr;
    IdxType* dptr_ = nullptr;
    IdxType* didx_ = nullptr;
#ifdef HAVE_MKL_
    sparse_matrix_t mkl_matrix_;
    struct matrix_descr mkl_descr_; 
    void MKLInit() {
        if (std::is_floating_point<ValType>::value) {
            mkl_descr_.type = SPARSE_MATRIX_TYPE_GENERAL;
            mkl_descr_.mode = SPARSE_FILL_MODE_LOWER;
            mkl_descr_.diag = SPARSE_DIAG_NON_UNIT;
            mkl_create_csr(&mkl_matrix_, SPARSE_INDEX_BASE_ZERO,
                            rows_, cols_, rowptr_, rowptr_ + 1, colidx_, values_);
        }
    }
#endif

    void Allocate() {
        RT_CHECK(rows_ > 0);
        RT_CHECK(nnz_ > 0);
        RT_CHECK(nnz_ <= rows_ * cols_);
        rowptr_ = MMUtils::Alloc<IdxType>(rows_ + 1);
        colidx_ = MMUtils::Alloc<IdxType>(nnz_);
        values_ = MMUtils::Alloc<ValType>(nnz_);
        dptr_ = MMUtils::Alloc<IdxType>(rows_ + 1); 
        didx_ = MMUtils::Alloc<IdxType>(rows_);
        RT_CHECK(rowptr_);
        RT_CHECK(colidx_);
        RT_CHECK(values_);
        RT_CHECK(dptr_);
        RT_CHECK(didx_);
#ifdef HAVE_MKL_
        MKLInit();
#endif
    }

    void Destroy() {
        if (!wrapped_) {
            MMUtils::Free(rowptr_);
            MMUtils::Free(colidx_);
            MMUtils::Free(values_);
            MMUtils::Free(dptr_);
            MMUtils::Free(didx_);
        }

    }
}; // class CSR

#endif // CSR_H_