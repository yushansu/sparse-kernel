#ifndef RND_STREAM_H_
#define RND_STREAM_H_

#include <cstdint>
#include <mkl.h>
#include <cassert>

class RndStream {
  public:
    explicit RndStream(unsigned long long seed) {
        seed_ = seed;
        VSLStreamStatePtr vslstream;
        assert(vslNewStream(&vslstream, VSL_BRNG_SFMT19937, static_cast<MKL_UINT>(seed)) == VSL_STATUS_OK);
        stream_ = static_cast<void *>(vslstream);
    }

    virtual ~RndStream() {
        VSLStreamStatePtr vslstream = static_cast<VSLStreamStatePtr>(stream_);
        vslDeleteStream(&vslstream);
    }

    void Gaussianfp64(double mean, double sigma, size_t len, void* v) {
        VSLStreamStatePtr vslstream = (VSLStreamStatePtr)stream_;
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, vslstream, len, (double*)v, mean, sigma);
    }

    void Gaussianfp32(float mean, float sigma, size_t len, void* v) {
        VSLStreamStatePtr vslstream = (VSLStreamStatePtr)stream_;
        vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, vslstream, len, (float*)v, mean, sigma);
    }

    void Uniformfp64(double low, double high, size_t len,  void* v) {
        VSLStreamStatePtr vslstream = (VSLStreamStatePtr)stream_;
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, vslstream, len, (double*)v, low, high);
    }

    void Uniformfp32(float low, float high, size_t len,  void* v) {
        VSLStreamStatePtr vslstream = (VSLStreamStatePtr)stream_;
        vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, vslstream, len, (float*)v, low, high);
    }

    void Uniformint32(size_t len, void* v) {
        VSLStreamStatePtr vslstream = static_cast<VSLStreamStatePtr>(stream_);
        viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, vslstream, len, (unsigned int*)v);
    }

    void Uniformint64(size_t len, void* v) {
        VSLStreamStatePtr vslstream = (VSLStreamStatePtr)stream_;
        viRngUniformBits64(VSL_RNG_METHOD_UNIFORMBITS64_STD, vslstream, len, (unsigned long long*)v);
    }
        
  private:
    unsigned long long seed_;
    void *stream_;    
};

#endif // RND_STREAM_H_ 