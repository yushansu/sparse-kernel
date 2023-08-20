#ifndef EXCEPTION_H_
#define EXCEPTION_H_

#include <cstddef>
#include <string>
#include "utils.h"

class Error : public std::exception {
  private:
    std::string function_;
    std::string file_;
    uint32_t line_;
    std::string condition_;
    std::string msg_;
    
 public:
    Error(const char* function, const char* file, uint32_t line,
          const char* condition, const std::string& msg)
          : function_(function), line_(line),
            condition_(condition), msg_(msg) {
        file_ = detail::StripBasename(file);
    }

    /// Returns the complete error message, including the source location.
    const char* what() const noexcept override {
        std::string msg = detail::CombineStrings("[fail at ", file_, ":", line_,
                                                 "] \"", condition_, "\": ", msg_, "\n");
        return msg.c_str();
    }
};

#define THROW_ERROR(cond, msg) \
  throw Error(__func__, __FILE__, static_cast<uint32_t>(__LINE__), cond, msg)

#define RT_CHECK(cond, ...)                                            \
    if (!(cond)) {                                                  \
        THROW_ERROR(#cond, detail::CombineStrings(__VA_ARGS__));    \
    }

#endif // EXCEPTION_H_