#ifndef UTILS_H_
#define UTILS_H_

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <linux/mman.h>

namespace detail {

template<class T, typename... Args> class Singleton {
  public:
    static T* Instance(Args &&... args) {
        static T instance(args...);
        return &instance;
    }
  private:
    Singleton();
    ~Singleton();
    Singleton(const Singleton& s);
    Singleton& operator=(const Singleton& s);
};

template <class T>
inline bool CorrectnessCheck(const T *expected, const T *actual, size_t n,
                             double tol = 1e-6, double ignore_smaller_than_this = 0) {
    double norm = 0.0;
    double err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        err += (expected[i] - actual[i]) * (expected[i] - actual[i]);
        norm += expected[i] * expected[i]; 
    }
    if (norm == 0.0) {
        err = sqrt(err);
    } else {
        err = sqrt(err/norm);
    }
    if (err > tol) {
        std::cout << "FAILED " << err << std::endl;
        return false;
    } else {
        return true;
    }
}

template <typename K, typename V>
inline void SortDict(K *idx, V *w, size_t begin, size_t end) {
    if (begin >= end) {
        return;
    }
    size_t length = end - begin - 1;
    std::swap(idx[begin], idx[begin + length / 2]);
    std::swap(w[begin], w[begin + length / 2]);
    size_t last = begin;
    for (size_t i = begin + 1; i < end; i++) {
        if (idx[i] < idx[begin]) {
            ++last;
            std::swap(idx[last], idx[i]);
            std::swap(w[last], w[i]);
        }
    }
    std::swap(idx[begin], idx[last]);
    std::swap(w[begin], w[last]);
    SortDict<K, V>(idx, w, begin, last);
    SortDict<K, V>(idx, w, last + 1, end);
}

//! Helper function used to stack msg
inline std::ostream& _str(std::ostream& ss) {
    return ss;
}

template <typename T>
inline std::ostream& _str(std::ostream& ss, const T& t) {
    ss << t;
    return ss;
}

template <typename T, typename... Args>
inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args) {
    return _str(_str(ss, t), args...);
}

// Convert a list of string-like arguments into a single string.
template <typename... Args>
inline std::string CombineStrings(const Args&... args) {
    std::ostringstream ss;
    _str(ss, args...);
    return ss.str();
}

// Specializations for already-a-string types.
template <>
inline std::string CombineStrings(const std::string& str) {
    return str;
}

inline std::string CombineStrings(const char* c_str) {
    return c_str;
}

inline std::string StripBasename(const std::string& full_path) {
    const char kSeparator = '/';
    size_t pos = full_path.rfind(kSeparator);
    if (pos != std::string::npos) {
        return full_path.substr(pos + 1, std::string::npos);
    } else {
        return full_path;
    }
}

inline std::string RepeatString(const std::string &s, size_t n) { 
    std::string ret = s; 
    for (size_t i = 1; i < n; ++i) {
        ret += s;
    }
    return ret; 
}

} //namespace detail

#endif // UTILS_H_