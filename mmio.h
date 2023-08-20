#ifndef MMIO_H_
#define MMIO_H_

#define MM_MAX_LINE_LENGTH 1025
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64

typedef char MM_typecode[4];

/********************* MM_typecode query functions ***************************/
#define mm_is_matrix(typecode)            ((typecode)[0]=='M')
#define mm_is_sparse(typecode)            ((typecode)[1]=='C')
#define mm_is_coordinate(typecode)        ((typecode)[1]=='C')
#define mm_is_dense(typecode)             ((typecode)[1]=='A')
#define mm_is_array(typecode)             ((typecode)[1]=='A')
#define mm_is_complex(typecode)           ((typecode)[2]=='C')
#define mm_is_real(typecode)              ((typecode)[2]=='R')
#define mm_is_pattern(typecode)           ((typecode)[2]=='P')
#define mm_is_integer(typecode)           ((typecode)[2]=='I')
#define mm_is_symmetric(typecode)         ((typecode)[3]=='S')
#define mm_is_general(typecode)           ((typecode)[3]=='G')
#define mm_is_skew(typecode)              ((typecode)[3]=='K')
#define mm_is_hermitian(typecode)         ((typecode)[3]=='H')

/********************* MM_typecode modify functions ***************************/
#define mm_set_matrix(typecode)           ((*typecode)[0]='M')
#define mm_set_coordinate(typecode)       ((*typecode)[1]='C')
#define mm_set_array(typecode)            ((*typecode)[1]='A')
#define mm_set_dense(typecode)            mm_set_array(typecode)
#define mm_set_sparse(typecode)           mm_set_coordinate(typecode)
#define mm_set_complex(typecode)          ((*typecode)[2]='C')
#define mm_set_real(typecode)             ((*typecode)[2]='R')
#define mm_set_pattern(typecode)          ((*typecode)[2]='P')
#define mm_set_integer(typecode)          ((*typecode)[2]='I')
#define mm_set_symmetric(typecode)        ((*typecode)[3]='S')
#define mm_set_general(typecode)          ((*typecode)[3]='G')
#define mm_set_skew(typecode)             ((*typecode)[3]='K')
#define mm_set_hermitian(typecode)        ((*typecode)[3]='H')
#define mm_clear_typecode(typecode)       ((*typecode)[0]=(*typecode)[1]=(*typecode)[2]=' ', (*typecode)[3]='G')
#define mm_initialize_typecode(typecode)  mm_clear_typecode(typecode)

#define MM_MTX_STR         "matrix"
#define MM_ARRAY_STR       "array"
#define MM_DENSE_STR       "array"
#define MM_COORDINATE_STR  "coordinate" 
#define MM_SPARSE_STR      "coordinate"
#define MM_COMPLEX_STR     "complex"
#define MM_REAL_STR        "real"
#define MM_INT_STR         "integer"
#define MM_GENERAL_STR     "general"
#define MM_SYMM_STR        "symmetric"
#define MM_HERM_STR        "hermitian"
#define MM_SKEW_STR        "skew-symmetric"
#define MM_PATTERN_STR     "pattern"

std::string mm_typecode_to_str(MM_typecode matcode) {
    std::string str;
    if (mm_is_matrix(matcode)) {
        str += MM_MTX_STR;
    } else {
        return "ERROR";
    }

    str += " ";
    if (mm_is_sparse(matcode)) {
        str += MM_SPARSE_STR;
    } else if (mm_is_dense(matcode)) {
        str += MM_DENSE_STR;
    } else {
        return "ERROR";
    }

    str += " ";
    if (mm_is_real(matcode)) {
        str += MM_REAL_STR;
    } else if (mm_is_complex(matcode)) {
        str += MM_COMPLEX_STR;
    } else if (mm_is_pattern(matcode)) {
        str += MM_PATTERN_STR;
    } else if (mm_is_integer(matcode)) {
        str += MM_INT_STR;
    } else {
        return "ERROR";
    }

    str += " ";
    if (mm_is_general(matcode)) {
        str += MM_GENERAL_STR;
    } else if (mm_is_symmetric(matcode)) {
        str += MM_SYMM_STR;
    } else if (mm_is_hermitian(matcode)) {
        str += MM_HERM_STR;
    } else if (mm_is_skew(matcode)) {
        str += MM_SKEW_STR;
    } else {
        return "ERROR";
    }

    return str;
}

struct MTXUtils {
    enum Status {
        MM_OK = 10,
        MM_COULD_NOT_READ_FILE = 11,
        MM_PREMATURE_EOF = 12,
        MM_NOT_MTX = 13,
        MM_NO_HEADER = 14,
        MM_UNSUPPORTED_TYPE = 15,
        MM_LINE_TOO_LONG = 16,
        MM_COULD_NOT_WRITE_FILE = 17,
        MM_ERROR_DATA = 18
    };
    
    static Status MMReadHeader(std::ifstream &fin, MM_typecode *matcode,
                               size_t *rows, size_t *cols, size_t *lines) {
        if (fin.eof()) {
            return Status::MM_PREMATURE_EOF;
        }
        std::string banner;
        std::string mtx;
        std::string crd;
        std::string data_type;
        std::string storage_scheme;
        fin >> banner >> mtx >> crd >> data_type >> storage_scheme;
        // RT_CHECK for banner
        if(banner.compare(MatrixMarketBanner) != 0) {
            return Status::MM_NO_HEADER;
        }
        // first field
        if(mtx.compare(MM_MTX_STR) != 0) {
            return Status::MM_UNSUPPORTED_TYPE;
        }
        mm_set_matrix(matcode);
        // second field
        if (crd.compare(MM_SPARSE_STR) == 0) {
            mm_set_sparse(matcode);
        } else if (crd.compare(MM_DENSE_STR) == 0) {
            mm_set_dense(matcode);
        } else {
            return Status::MM_UNSUPPORTED_TYPE;
        }
        // third field
        if (data_type.compare(MM_REAL_STR) == 0) {
            mm_set_real(matcode);
        } else if (data_type.compare(MM_COMPLEX_STR) == 0) {
            mm_set_complex(matcode);
        } else if (data_type.compare(MM_PATTERN_STR) == 0) {
            mm_set_pattern(matcode);
        } else if (data_type.compare(MM_INT_STR) == 0) {
            mm_set_integer(matcode);
        } else {
            return Status::MM_UNSUPPORTED_TYPE;
        }
        // fourth field
        if (storage_scheme.compare(MM_GENERAL_STR) == 0) {
            mm_set_general(matcode);
        } else if (storage_scheme.compare(MM_SYMM_STR) == 0) {
            mm_set_symmetric(matcode);
        } else if (storage_scheme.compare(MM_HERM_STR) == 0) {
            mm_set_hermitian(matcode);
        } else if (storage_scheme.compare(MM_SKEW_STR) == 0) {
            mm_set_skew(matcode);
        } else {
            return Status::MM_UNSUPPORTED_TYPE;
        }
        // read matrix size
        if (fin.eof()) {
            return Status::MM_PREMATURE_EOF;
        }
        fin >> *rows >> *cols >> *lines;
        return MM_OK;
    }

    template <typename IdxType, typename ValType>
    static Status MMReadMtxCrd(std::ifstream &fin, MM_typecode matcode,
                               size_t rows, size_t cols, size_t lines,
                               size_t *nnz, IdxType *rowidx, IdxType *colidx, ValType *values,
                               bool is_symmetric) {
        // read values
        size_t count = 0;
        size_t rd_lines = 0;
        int base = 1;
        while (!fin.eof() && rd_lines < lines) {
            ValType val = ValType();
            IdxType x;
            IdxType y;
            ValType real;
            ValType imag;
            if (mm_is_pattern(matcode)) {
                fin >> x  >> y;
            } else if (mm_is_complex(matcode)) {
                fin >> x  >> y >> real >> imag;
                val = real + imag;
            } else {
                fin >> x  >> y >> val;
            }
            if (x > rows || y > cols) {
                return Status::MM_ERROR_DATA;
            }
            if (0 == x || 0 == y) {
                base = 0;
            }
            rowidx[count] = x;
            colidx[count] = y;
            if (!mm_is_pattern(matcode)) {
                values[count] = val;
            }
            rd_lines++;
            count++;
            if (is_symmetric && x != y) {
                rowidx[count] = y;
                colidx[count] = x;
                if (!mm_is_pattern(matcode)) {
                    values[count] = val;
                }
                count++;
            }
        }
        if (lines > rd_lines) {
            return Status::MM_PREMATURE_EOF;
        }
        *nnz = count;
        if (1 == base) {
            for (size_t i = 0; i < *nnz; ++i) {
                rowidx[i]--;
                colidx[i]--;
            }
        }
        return Status::MM_OK;
    }

    static void MMWriteHeader(std::ofstream &fout, size_t rows, size_t cols,
                              size_t nnz, bool pattern = false) {
        MM_typecode matcode;
        mm_initialize_typecode(&matcode);
        mm_set_matrix(&matcode);
        mm_set_sparse(&matcode);
        if (pattern) {
            mm_set_pattern(&matcode);
        } else {
            mm_set_real(&matcode);
        }
        // print banner followed by typecode.
        fout << MatrixMarketBanner << " " << mm_typecode_to_str(matcode) << std::endl;
        fout << rows << " " << cols << " " << nnz << std::endl;
    }
};

# endif