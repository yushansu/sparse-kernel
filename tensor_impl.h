#ifndef TENSOR_IMPL_H_
#define TENSOR_IMPL_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include "utils.h"
#include "exception.h"

namespace spdt {

enum NormType {
  NORM_1,
  NORM_2,
  NORM_INF,
  NORM_FBS,
};

// forward declarations

template <int d> struct TensorPrinter;

template <int d, class Tensor> struct TensorProxy;

// helper classes

//! type 2 type construct
template <class T> class Type2Type { typedef T OriginalType; };

//! Class construct for different classes
template <class A, class B> struct SameClass {
    enum { result = false };
};

//! Class construct for the same class
template <class A> struct SameClass<A, A> {
    enum { result = true };
};


////////////////////////////////////////////////////////////////////////////////
// iterator classes

template <typename T, typename P, int d = -1> class TensorIterator;

//! Tensor iterator class template
/*! This class is used to iterate over any dimension of the tensor.
 * The class contains as data members a pointer to memory, and the
 * stride that will be used to advance the iterator.
 */
template <typename T, typename P, int d>
class TensorIterator : public std::iterator<std::random_access_iterator_tag, T,
                                            std::ptrdiff_t, P> {
  public:
    typedef P Pointer;
        
    //! Pointer and stride parameter constructor
    TensorIterator(T *x, size_t stride) : p_(x), stride_(stride) {}
    
    //! Copy constructor
    TensorIterator(const TensorIterator &other) : p_(other.p_), stride_(other.stride_) {}
    
    //! Pre-increment iterator
    TensorIterator &operator++() {
        p_ += stride_;
        return *this;
    }

    //! Post-increment iterator
    TensorIterator operator++(int) {
        TensorIterator it(p_);
        p_ += stride_;
        return it;
    }

    //! Pre-decrement iterator
    TensorIterator &operator--() {
        p_ -= stride_;
        return *this;
    }

    //! Post-decrement iterator
    TensorIterator operator--(int) {
        TensorIterator it(p_);
        p_ -= stride_;
        return it;
    }

    //! Equal-to operator
    bool operator==(const TensorIterator &rhs) {
        return p_ == rhs.p_;
    }

    //! Not-equal-to operator
    bool operator!=(const TensorIterator &rhs) {
        return p_ != rhs.p_;
    }

    //! DeReference operator
    T &operator*() { return *p_; }

  private:
    Pointer p_;  //!< Pointer to memory
    size_t stride_; //!< Stride
};

//! Tensor iterator class template
/*! This partial template specialization is used to iterate over all
 * the elements of the tensor.
 */
template <typename T, typename P>
class TensorIterator<T, P, -1>
    : public std::iterator<std::random_access_iterator_tag, T, std::ptrdiff_t, P> {

  public:
    typedef P Pointer;

    //! Pointer parameter constructor
    TensorIterator(T *x) : p_(x) {}

    //! Copy constructor
    TensorIterator(const TensorIterator &other) : p_(other.p_) {}

    //! Pre-increment iterator
    TensorIterator &operator++() {
        ++p_;
        return *this;
    }

    //! Post-increment iterator
    TensorIterator operator++(int) {
        return TensorIterator(p_++);
    }

    //! Pre-decrement iterator
    TensorIterator &operator--() {
        --p_;
        return *this;
    }

    //! Post-decrement iterator
    TensorIterator operator--(int) {
        return TensorIterator(p_--);
    }

    //! Equal-to operator
    bool operator==(const TensorIterator &rhs) {
        return p_ == rhs.p_;
    }

    //! Not-equal-to operator
    bool operator!=(const TensorIterator &rhs) {
        return p_ != rhs.p_;
    }

    //! DeReference operator
    T &operator*() { return *p_; }

  private:
    Pointer p_; //!< Pointer to memory
};

////////////////////////////////////////////////////////////////////////////////
// TensorTraits clases

//! TensorTraits class
template <int k, typename T, class TensorType> class TensorTraits {};

//! TensorTraits partial template specialization for vectors
template <typename T, class TensorType> class TensorTraits<1, T, TensorType> {

  protected:
    typedef T ValType;

    //! Helper function used to fill a vector from a functor, lambda expression,
    // etc.
    template <class functor> void Fill(functor fn, ValType *data = nullptr) {
        TensorType &a = static_cast<TensorType &>(*this);
        if (data) {
            for (size_t i = 0; i < a.n_[0]; ++i) {
                data[i] = fn(i);
            }
        } else {
            for (size_t i = 0; i < a.n_[0]; ++i) {
                a.data_[i] = fn(i);
            }
        }
    }

  public:
#if 0
    /*! \brief Obtain the norm of a vector
     *
     * This function calls a helper function depending on the type stored in the
     * vector.
     */
    template <typename ValType>
    typename std::enable_if<std::is_floating_point<ValType>::value, ValType>::type
    Norm(NormType n = NORM_2) const {
        return Norm(n, Type2Type<ValType>());
    }

    /*! \brief Normalize vector to unit length
     */
    template <typename ValType>
    typename std::enable_if<!std::is_floating_point<ValType>::value, TensorType>::type
    &Normalize() {
        TensorType &a = static_cast<TensorType &>(*this);
        ValType norm = Norm();
        for (size_t i = 0; i < a.n_[0]; ++i) {
            a.data_[i]= a.data_[i]/norm;
        }
        return a;
    }
#endif
    size_t length() const {
        return n_[0];
    }

  private:
#if 0
    //! Helper function used by norm
    template <typename U>
    inline typename std::enable_if<std::is_floating_point<U>::value, U>::type
    Norm(NormType n, Type2Type<U>) const {
        U norm = U();
        const TensorType &a = static_cast<const TensorType &>(*this);
        switch (n) {
        case NORM_1:
            for (size_t i = 0; i < a.n_[0]; ++i) {
                norm += fabs(a.data_[i]);
            }
            break;
        case NORM_2:
            for (size_t i = 0; i < a.n_[0]; ++i) {
                norm += std::pow(a.data_[i], 2);
            }
            norm = std::sqrt(norm);
        case NORM_INF: {
            for (size_t i = 0; i < a.n_[0]; ++i) {
                U tmp = fabs(a.data_[i]);
                norm = std::max(norm, tmp);
            }
            break;
        }
        default:
            std::cout << "Error: " << n << " not implemented for matrices" << std::endl;
            exit(1);
        }
        return norm;
    }
#endif
};

//! TensorTraits partial template specialization for matrices
template <typename T, class TensorType> class TensorTraits<2, T, TensorType> {

  protected:
    typedef T ValType;

    typedef std::initializer_list<T> list_type;

    //! Helper function used to fill a vector from a functor, lambda expression,
    // etc.
    template <class functor> void Fill(functor fn, ValType *data = nullptr) {
        TensorType &a = static_cast<TensorType &>(*this);
        if (data) {
            for (size_t r = 0; r < a.n_[0]; ++r) {
                for (size_t c = 0; c < a.n_[1]; ++c) {
                    data[r + c * a.ldn_[1]] = fn(r, c);
                }
            }
        } else {
            for (size_t r = 0; r < a.n_[0]; ++r) {
                for (size_t c = 0; c < a.n_[1]; ++c) {
                    a.data_[r + c * a.ldn_[1]] = fn(r, c);
                }
            }
        }
    }

  public:
#if 0
    //! Matrix norm
    /*! This function calls a helper function depending on the type
     * stored in the matrix.
     */
    template <typename ValType>
    typename std::enable_if<!std::is_floating_point<ValType>::value, ValType>::type
    Norm(NormType n = NORM_FBS) const {
        return Norm(n, Type2Type<ValType>());
    }
#endif

    size_t rows() const {
        const TensorType &a = static_cast<const TensorType &>(*this);
        return a.n_[0];
    }

    size_t cols() const {
        const TensorType &a = static_cast<const TensorType &>(*this);
        return a.n_[1];
    }

    size_t ldr() const {
        const TensorType &a = static_cast<const TensorType &>(*this);
        return a.ldn_[0];
    }

    size_t ldc() const {
        const TensorType &a = static_cast<const TensorType &>(*this);
        return a.ldn_[1];
    }
    
  private:
#if 0
    //! Helper function used by norm
    template <typename U>
    inline typename std::enable_if<std::is_floating_point<U>::value, U>::type
    Norm(NormType n, Type2Type<U>) const {
        U norm = U();
        const TensorType &a = static_cast<const TensorType &>(*this);
        switch (n) {
        case NORM_1: {
            // loop over columns
            for (size_t c = 0; c < a.n_[1]; ++c) {
                U tmp = U();
                for (size_t r = 0; r < a.n_[0]; ++r) {
                    tmp += a.data_[r + c * a.ldn_[1]];
                }
                norm = std::max(tmp, norm);
            }
            break;
        }
        case NORM_INF: {
            // loop over rows
            for (size_t r = 0; r < a.n_[0]; ++r) {
                U tmp = U();
                for (size_t c = 0; c < a.n_[1]; ++c) {
                    tmp += a.data_[r + c * a.ldn_[1]];
                }
                norm = std::max(tmp, norm);
            }
            break;
        }
        case NORM_FBS: {
            // loop over rows
            for (size_t r = 0; r < a.n_[0]; ++r) {
                U tmp = U();
                for (size_t c = 0; c < a.n_[1]; ++c) {
                    tmp += std:pow(a.data_[r + c * a.ldn_[1]], 2.0);
                }
                norm = std::sqrt(tmp);
            }
            break;
        }
        default:
            std::cout << "Error: " << n << " not implemented for matrices" << std::endl;
            exit(1);
        }
        return norm;
    }
#endif
};

//! Tensor traits partial template specialization for 3rd order tensors
template <typename T, class TensorType> class TensorTraits<3, T, TensorType> {

  protected:
    typedef T ValType;

    //! Helper function used to fill a vector from a functor, lambda expression,
    // etc.
    template <class functor> void Fill(functor fn, ValType *data = nullptr) {
        TensorType &a = static_cast<TensorType &>(*this);
        if (data) {
            for (size_t i = 0; i < a.n_[0]; ++i) {
                for (size_t j = 0; j < a.n_[1]; ++j) {
                    for (size_t k = 0; k < a.n_[2]; ++k) {
                        data[i * a.ldn_[0] + j * a.ldn_[1] + k * a.ldn_[2]] = fn(i, j, k);
                    }
                }
            }
        } else {
            for (size_t i = 0; i < a.n_[0]; ++i) {
                for (size_t j = 0; j < a.n_[1]; ++j) {
                    for (size_t k = 0; k < a.n_[2]; ++k) {
                        a.data_[i * a.ldn_[0] + j * a.ldn_[1] + k * a.ldn_[2]] = fn(i, j, k);
                    }
                }
            }
        }
    }
};

//! Tensor traits partial template specialization for 4th order tensors
template <typename T, class TensorType> class TensorTraits<4, T, TensorType> {

  protected:
    typedef T ValType;

    //! Helper function used to fill a vector from a functor, lambda expression,
    // etc.
    template <class functor> void Fill(functor fn, ValType *data = nullptr) {
        TensorType &a = static_cast<TensorType &>(*this);
        if (data) {
            for (size_t i = 0; i < a.n_[0]; ++i) {
                for (size_t j = 0; j < a.n_[1]; ++j) {
                    for (size_t k = 0; k < a.n_[2]; ++k) {
                        for (size_t l = 0; l < a.n_[3]; ++l) {
                            data[i * a.ldn_[0] + j * a.ldn_[1] + k * a.ldn_[2] + l * a.ldn_[3]] =
                                fn(i, j, k, l);
                        }
                    }
                }
            }
        } else {
            for (size_t i = 0; i < a.n_[0]; ++i) {
                for (size_t j = 0; j < a.n_[1]; ++j) {
                    for (size_t k = 0; k < a.n_[2]; ++k) {
                        for (size_t l = 0; l < a.n_[3]; ++l) {
                            a.data_[i * a.ldn_[0] + j * a.ldn_[1] + k * a.ldn_[2] + l * a.ldn_[3]] =
                                fn(i, j, k, l);
                        }
                    }
                }
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////
// tensor class template

//! Tensor class template
/*! This class template is used to define tensors of any rank.
 * \tparam k - Tensor rank
 * \tparam T - Type stored in the tensor
 *
 * The class template inherits from TensorTraits passing itself as
 * a template parameter to use the Curiously Recurring Template
 * Pattern (CRTP).
 */
template <int k, typename T>
class Tensor : public TensorTraits<k, T, Tensor<k, T>> {
    static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
                  "Not supported");
  public:
    typedef T ValType;
    typedef T *Pointer;
    typedef const T *ConstPointer;
    typedef T &Reference;
    typedef const T &ConstReference;

    typedef TensorIterator<ValType, Pointer> Iterator;
    typedef TensorIterator<const ValType, ConstPointer> ConstIterator;
    typedef std::reverse_iterator<Iterator> ReverseIterator;
    typedef std::reverse_iterator<ConstIterator> ConstReverseIterator;

  private:
    size_t n_[k] = {0};       //!< Tensor dimensions
    size_t ldn_[k] = {0};     //!< Tensor leading dimensions
    size_t size_ = 0;         //!< Tensor total size
    size_t ldsize_ = 0;         //!< Tensor total size
    Pointer data_ = nullptr;  //!< Pointer to memory
    bool wrapped_ = false;    //!< Owned memory flag
    bool hugepage_ = false;
    bool hugepage_1g_ = false;

  public:
    //! Rank of the tensor
    constexpr static int rank() { return k; }

    //! Pointer to memory for raw access
    Pointer data() const {
        return data_;
    }

    //! Copy constructor
    Tensor(const Tensor &a);

    //! Move constructor
    Tensor(Tensor &&other);

    //! Assignment operator
    Tensor &operator=(const Tensor &other);

    //! Move assignment operator
    Tensor &operator=(Tensor &&other);

    //! Destructor
    ~Tensor() {
        Destroy();
    }

    void ToHugePage() {
        if (!hugepage_) {
            ValType *data = data_;
            data_ = HP_ALLOC(ValType, ldsize_);
            RT_CHECK(data_);
            std::copy_n(data, ldsize_, data_);
            if (hugepage_1g_) {
                HP_1G_FREE(data, ldsize_)
            } else {
                MMUtils::Free(data);
            }
            hugepage_ = true;
        }
    }

    void ToHugePage1G() {
        if (!hugepage_1g_) {
            ValType *data = data_;
            data_ = HP_1G_ALLOC(ValType, ldsize_);
            RT_CHECK(data_);
            std::copy_n(data, ldsize_, data_);
            if (hugepage_) {
                HP_FREE(data, ldsize_)
            } else {
                MMUtils::Free(data);
            }
            hugepage_1g_ = true;
        }
    }

  private:
    void Allocate() {
        data_ = MMUtils::Alloc<ValType>(ldsize_);
        RT_CHECK(data_);
    }

    void Destroy() {
        if (!wrapped_) {
            MMUtils::Free(data_);
        }
    }

    //! Helper function used by constructors
    void InitDim() {
        if (n_[0] == 0) {
            n_[0] = 1;
        }
        size_ = n_[0];
        for (size_t i = 1; i < k; ++i) {
            if (n_[i] == 0) {
                n_[i] = n_[i - 1];
            }
            size_ *= n_[i];
        }
        ldn_[0] = 1;
        for (size_t i = 1; i < k; ++i) {
            ldn_[i] = ldn_[i - 1] * n_[i - 1]; 
        }
        ldsize_ = ldn_[k - 1] * n_[k - 1];
    }

    //! init helper function that takes an integer parameter
    template <int d, typename U, typename... Args>
    typename std::enable_if<std::is_integral<U>::value and !std::is_pointer<U>::value and d < k, void>::type
    Init(U i, Args &&... args) {
        RT_CHECK(i != 0);
        n_[d] = i;
        Init<d + 1>(args...);
    }

    //! init helper function that takes a value to initialize all elements
    template <int d>
    void Init(ValType v = ValType()) {
        InitDim();
        data_ = MMUtils::Alloc<ValType>(ldsize_);
        RT_CHECK(data_);
        std::fill_n(data_, ldsize_, v);
    }

    //! init helper function that takes a pointer to already existing data
    template <int d, typename... Args>
    void Init(Pointer p) {
        InitDim();
        wrapped_ = true;
        data_ = p;
    }

    //! init helper function that takes a functor, lambda expression, etc.
    template <int d, class functor>
    typename std::enable_if<!std::is_integral<functor>::value and
                            !std::is_pointer<functor>::value and
                            !std::is_floating_point<functor>::value, void>::type
    Init(functor fn) {
        InitDim();
        Allocate();
        this->Fill(fn, data_);
    }

    double RandHelper() {
        return drand48();
    }

  public:
    Tensor() {};
    
    Tensor(const size_t *n, const size_t *pad_n = nullptr) {
        ldn_[0] = 1;
        size_ = 1;
        ldsize_ = 1;
        for (size_t i = 0; i < k; ++i) {
            RT_CHECK(n[i] > 0);
            RT_CHECK(n[i] <= pad_n[i]);
            n_[i] = n[i];
            if (i > 0) {
                ldn_[i] = ldn_[i - 1] * n_[i - 1];
            }
            size_ *= n_[i];
            ldsize_ *= pad_n[i];
        }
        Allocate();
    }

    //! Parameter constructor, uses the init helper function
    template <typename... Args>
    explicit Tensor(const Args &... args) {
        static_assert(sizeof...(Args) <= k + 2,
                      "*** ERROR *** Wrong number of arguments for Tensor");
        static_assert(sizeof...(Args) >= k,
                      "*** ERROR *** Wrong number of arguments for Tensor");
        Init<0>(args...);
    }

    //! Helper structure used to process initializer lists
    template <int d, typename U> struct InitializerList {
        typedef std::initializer_list<typename InitializerList<d - 1, U>::list_type> list_type;
        static void process(list_type l, Tensor &a, size_t s, size_t idx) {
            a.n_[k - d] = l.size(); // set dimension
            size_t j = 0;
            for (const auto &r : l) {
                InitializerList<d - 1, U>::process(r, a, s * l.size(), idx + s * j++);
            }
        }
    };

    //! Helper structure used to process initializer lists, partial template
    // specialization to finish recursion
    template <typename U> struct InitializerList<1, U> {
        typedef std::initializer_list<U> list_type;
        static void process(list_type l, Tensor &a, size_t s, size_t idx) {
            a.n_[k - 1] = l.size(); // set dimension
            if (!a.data_) {
                a.data_ = new ValType[s * l.size()];
            }
            size_t j = 0;
            for (const auto &r : l) {
                a.data_[idx + s * j++] = r;
            }
        }
    };

    typedef typename InitializerList<k, T>::list_type InitializerType;

    //! Initializer list constructor
    explicit Tensor(InitializerType l) : data_(nullptr), wrapped_() {
        InitializerList<k, T>::process(l, *this, 1, 0);
        InitDim();
    }

    //! Size of the tensor
    size_t size() const {
        return size_;
    }

    size_t ldsize() const {
        return ldsize_;
    }

    //! Size along the ith direction
    size_t dim(size_t i) const {
        assert(i >=0 && i < k);
        return n_[i];
    }

    //! Leading dim of the tensor
    size_t ldim(size_t i) const {
        assert(i >=0 && i < k);
        return ldn_[i];
    }

    void RandInit(ValType low = 0, ValType high = 1) {
        assert(low <= high);   
        assert(low <= high);
        for (size_t i = 0; i < ldsize_; ++i) {
            data_[i] = static_cast<ValType>(RandHelper() * (high - low)) + low;
        }
    }

    void ValueInit(ValType v = ValType()) {
        for (size_t i = 0; i < ldsize_; ++i) {
            data_[i] = v;
        }
    }

  private:
    ////////////////////////////////////////////////////////////////////////////////
    // indexed access operators

    //! Helper function used to compute the index on the one-dimensional tensor
    // that stores the tensor elements
    template <typename PackType> PackType index(PackType indices[]) const {
        PackType idx = indices[0] * ldn_[0];
        for (size_t i = 1; i < k; ++i) {
            assert(indices[i] >= 0);
            assert(static_cast<size_t>(indices[i]) < n_[i]);
            idx += indices[i] * ldn_[i];
        }
        return idx;
    }

    //! Helper structure used by operator()
    template <typename first_type, typename... Rest> struct RT_CHECK_integral {
        typedef first_type PackType;
        enum { tmp = std::is_integral<first_type>::value };
        enum { value = tmp && RT_CHECK_integral<Rest...>::value };
        static_assert(value, "*** ERROR *** Non-integral type parameter found.");
    };

    //! Partial template specialization that finishes the recursion, or
    // specialization used by vectors.
    template <typename last_type> struct RT_CHECK_integral<last_type> {
        typedef last_type PackType;
        enum { value = std::is_integral<last_type>::value };
    };

  public:
    //! Indexed access through operator()
    template <typename... Args> Reference operator()(Args... params) {
        // RT_CHECK that the number of parameters corresponds to the size of the
        // tensor
        static_assert(sizeof...(Args) == k,
            "*** ERROR *** Number of parameters does not match tensor rank.");
        typedef typename RT_CHECK_integral<Args...>::PackType PackType;
        // unpack parameters
        PackType indices[] = {params...};
        // return Reference
        return data_[index(indices)];
    }

    //! Indexed access through operator() for constant tensors
    template <typename... Args> ValType operator()(Args... params) const {
        // RT_CHECK that the number of parameters corresponds to the size of the
        // tensor
        static_assert(sizeof...(Args) == k,
            "*** ERROR *** Number of parameters does not match tensor rank.");
        typedef typename RT_CHECK_integral<Args...>::PackType PackType;
        // unpack parameters
        PackType indices[] = {params...};
        // return Reference
        return data_[index(indices)];
    }

    //! Helper function used to compute the stride of iterators
    size_t stride(size_t dim) const {
        size_t s = 1;
        for (int j = 0; j < dim; ++j) {
            s *= n_[j];
        }
        return s;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // iterator functions

    //! Iterator begin
    Iterator begin() { return Iterator(data_); }

    //! Iterator begin for constant tensors
    ConstIterator begin() const { return ConstIterator(data_); }

    //! Iterator end
    Iterator end() { return Iterator(data_ + size()); }

    //! Iterator end for constant tensors
    ConstIterator end() const { return ConstIterator(data_ + size()); }

    //! Reverse iterator begin
    ReverseIterator rbegin() { return ReverseIterator(end()); }

    //! Reverse iterator begin for constante tensors
    ConstReverseIterator rbegin() const {
        return ConstReverseIterator(end());
    }

    //! Reverse iterator end
    ReverseIterator rend() { return ReverseIterator(begin()); }

    //! Reverse iterator end for constante tensors
    ConstReverseIterator rend() const {
        return ConstReverseIterator(begin());
    }

    ////////////////////////////////////////////////////////////////////////////////
    // dimensional iterator functions

    template <int d> using dIterator = TensorIterator<ValType, Pointer, d>;

    //! Dimensional iterator begin
    template <int d> dIterator<d> dbegin() {
        return dIterator<d>(data_, stride(d));
    }

    //! Dimensional iterator end
    template <int d> dIterator<d> dend() {
        size_t s = stride(d);
        return dIterator<d>(data_ + stride(d + 1), s);
    }

    //! Dimensional iterator begin
    template <int d, typename Iterator> dIterator<d> dbegin(Iterator it) {
        return dIterator<d>(&*it, stride(d));
    }

    //! Dimensional iterator end
    template <int d, typename Iterator> dIterator<d> dend(Iterator it) {
        size_t s = stride(d);
        return dIterator<d>(&*it + stride(d + 1), s);
    }

    //! Dimensional iterator begin for constant tensors
    template <int d> dIterator<d> dbegin() const {
        return dIterator<d>(data_, stride(d));
    }

    //! Dimensional iterator end for constant tensors
    template <int d> dIterator<d> dend() const {
        size_t s = stride(d);
        return dIterator<d>(data_ + stride(d + 1), s);
    }

    //! Dimensional iterator begin for constant tensors
    template <int d, typename Iterator> dIterator<d> dbegin(Iterator it) const {
        return dIterator<d>(&*it, stride(d));
    }

    //! Dimensional iterator end for constant tensors
    template <int d, typename Iterator> dIterator<d> dend(Iterator it) const {
        size_t s = stride(d);
        return dIterator<d>(&*it + stride(d + 1), s);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // friend classes and functions
    friend class TensorTraits<k, T, Tensor>;

    // make tensors of other ranks a friend class
    template <int s, typename S> friend struct Tensor;

    //! Standard output
    friend std::ostream &operator<<(std::ostream &os, const Tensor &a) {
        if (a.size() == 0) {
            os << "Empty tensor" << std::endl;
            return os;
        }
        if (a.wrapped_) {
            os << "Wrapped ";
        }
        TensorPrinter<k>::Print(os, a.n_, a.ldn_, a.data_);
        return os;
    }
}; // class Tensor

////////////////////////////////////////////////////////////////////////////////
// implementation of tensor functions

// copy constructor
template <int k, typename T>
Tensor<k, T>::Tensor(const Tensor<k, T> &other)
    : wrapped_(other.wrapped_), size_(other.size_), ldsize_(other.ldsize_) {
    std::copy_n(other.n_, k, n_);
    std::copy_n(other.ldn_, k, ldn_);
    if (!other.wrapped_) {
        if (other.size_ > 0) {
            Allocate();
            std::copy_n(other.data_, ldsize_, data_);
        } // if (other.size_ > 0)
    } else {
        data_ = other.data_;
    } // if (!other.wrapped_) 
}

// move constructor
template <int k, typename T>
Tensor<k, T>::Tensor(Tensor<k, T> &&other) {
    std::copy_n(other.n_, k, n_);
    std::copy_n(other.ldn_, k, ldn_);
    data_ = other.data_;
    wrapped_ = other.wrapped_;
    size_ = other.size_;
    ldsize_ = other.ldsize_;
    // set other to default
    std::fill_n(other.n_, k, 0);
    std::fill_n(other.ldn_, k, 0);
    other.data_ = nullptr;
    other.wrapped_ = false;
    other.size_ = 0;
    other.ldsize_ = 0;
}

// assignment operator
template <int k, typename T>
Tensor<k, T> &Tensor<k, T>::operator=(const Tensor<k, T> &other) {
    if (this != &other) {
        Destroy();
        std::copy_n(other.n_, k, n_);
        std::copy_n(other.ldn_, k, ldn_);
        wrapped_ = other.wrapped_;
        size_ = other.size_;
        ldsize_ = other.ldsize_;
        if (!other.wrapped_) { 
            if (other.size_ > 0) {
                std::copy_n(other.data_, ldsize_, data_);
            } // if (other.size_ > 0)
        } else {
            data_ = other.data_;
        } // if (!other.wrapped_)
    } // if (this != &other)
    return *this;
}

// move assignment operator
template <int k, typename T>
Tensor<k, T> &Tensor<k, T>::operator=(Tensor<k, T> &&other) {
    if (this != &other) {
        Destroy();
        std::copy_n(other.n_, k, n_);
        std::copy_n(other.ldn_, k, ldn_);
        wrapped_ = other.wrapped_;
        data_ = other.data_;
        size_ = other.size_;
        ldsize_ = other.ldsize_;
        // set other to default
        other.size_ = 0;
        other.ldsize_ = 0;
        other.data_ = nullptr;
        other.wrapped_ = false;
        std::fill_n(other.n_, k, 0);
        std::fill_n(other.ldn_, k, 0);
    }
    return *this;
}

//! Print template partial specialization for vectors
template <> struct TensorPrinter<1> {
    template <typename ValType>
    static std::ostream &Print(std::ostream &os, const size_t n[], const size_t ldn[], const ValType *data) {
        os << "Vector(" << n[0] << ")" << std::endl;
        os << " ";
        for (size_t i = 0; i < n[0]; ++i) {
            os << " " << data[i];
        }
        os << std::endl;
        return os;
    }
};

//! Print template partial specialization for matrices
template <> struct TensorPrinter<2> {
    template <typename ValType>
    static std::ostream &Print(std::ostream &os, const size_t n[], const size_t ldn[], const ValType *data) {
        os << "Matrix(" << n[0] << "x" << n[1] << ")" << std::endl;
        for (size_t r = 0; r < n[0]; ++r) {
            os << " ";
            for (size_t c = 0; c < n[1]; ++c) {
                os << " " << data[r * ldn[0] + c * ldn[1]];
            }
            os << std::endl;
        }
        return os;
    }
};

//! Print class template that is used for tensors of order greater than 2
template <int d> struct TensorPrinter {
    template <typename ValType>
    static std::ostream &Print(std::ostream &os, const size_t n[], const size_t ldn[], const ValType *data) {
        for (size_t i = 0; i < n[0]; ++i) {
            os << i << ", ";
            if (d - 1 == 2) {
                os << ":, :" << std::endl;
            }
            TensorPrinter<d - 1>::Print(os, ldsize, &n[1], &ldn[1], data + i * ldn[0]);
        }
        return os;
    }
};

} // namespace spdt

#endif // TENSOR_IMPL_H_