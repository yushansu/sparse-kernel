#include <immintrin.h>
#include <iostream>
#include <stdio.h>

class SimpleSpinLock{
  public:
    SimpleSpinLock()
        : state_(Free) {
        
    }
    
    void Lock() {
        while (__sync_val_compare_and_swap(&state_, Free, Busy) != Free) {
            do {
                _mm_pause();
            } while (state_ == Busy);
        }
    }
    
    void Unlock() {
        state_ = Free;
    }
    
    bool Locked() const {
        return state_ == Busy;
    }
  
  private:
    volatile unsigned int state_;
    enum {
        Free = 0,
        Busy = 1
    };
};

class TransactionScope
{
  public:
    TransactionScope(SimpleSpinLock& fallback_lock, int max_retries = 10, int tid = 0)
        : fallback_lock_(fallback_lock) {
        int num_retries = 0;
        while (1) {
            ++num_retries;
            unsigned status = _xbegin();
            if (status == _XBEGIN_STARTED) {
                if (!fallback_lock_.Locked()) {
                    return;
                }
                _xabort(0xff);
            }
            if ((status & _XABORT_EXPLICIT) && _XABORT_CODE(status)==0xff && !(status & _XABORT_NESTED)) {
                while (fallback_lock_.Locked()) {
                    _mm_pause();
                }
            }
            if (num_retries >= max_retries) {
                break;
            }
        }
        fallback_lock_.Lock();
    }

    virtual ~TransactionScope() {
        if (fallback_lock_.Locked()) {
            fallback_lock_.Unlock();
        } else {
            _xend();
        }
    }

  private:
    SimpleSpinLock & fallback_lock_;

    TransactionScope();
};