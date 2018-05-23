#include <stdint.h>

#define WARP_SIZE 32

// -------------------------------------------------------------------
// helper functions
// -------------------------------------------------------------------

// Get largest memory address that is aligned to a warp worth of floats
// and smaller than x.
__forceinline__ __device__ uintptr_t getBlockBeginning(void const * x)
{
  return (uintptr_t)(x) & (~((uintptr_t)(WARP_SIZE*sizeof(float)) - 1)) ;
}

// Call this kernel like compute_moments, but it does not need a scratch space
// DOCS
/*      beta = x_hat .* x_hat_der_sig ./ (sig_sq + epsilon) ;*/
/*      dx_hat = alpha - beta ;*/
/*      der = g .* dx_hat + x_hat .* dg + db ;*/
/*
//      dx_hat = alpha - x_hat .* x_hat_der_sig ./ (sig_sq + epsilon) ;
//      der = g .* dx_hat + x_hat .* dg + db ;*/

__global__ void mega_kernel(float* xHat,
                            float const * alpha,
                            float const * moments,
                            float const * xHatDerSig,
                            float const * gains,
                            float const * dGains,
                            float const * dBias,
                            float const epsilon,
                            int planeArea,
                            int numPlanes,
                            int numChannels)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;

  float sigma2 = moments[channel+numChannels] * moments[channel+numChannels] ;
  float x_hat_der_sig = xHatDerSig[channel] ;
  float g = gains[channel] ;
  float dg = dGains[channel] ;
  float db = dBias[channel] ;

  float coefficient = x_hat_der_sig / (sigma2 + epsilon) ;
  float dxHat ;

  while (plane < numPlanes) {
    float const * planeBegin = xHat + plane * planeArea ;
    float const * planeEnd = planeBegin + planeArea ;
    float const * xHatBlock = (float const*) getBlockBeginning(planeBegin) + tid ;

    float const * aPlaneBegin = alpha + plane * planeArea ;
    float const * aBlock = (float const*) getBlockBeginning(aPlaneBegin) + tid ;

    float * oblock = xHat + (xHatBlock - xHat) ;

    while (xHatBlock < planeEnd) {
      if (xHatBlock >= planeBegin) {
        dxHat = *aBlock - coefficient * (*xHatBlock) ;
        *oblock = g * dxHat + dg * (*xHatBlock) + db ;
      }
      xHatBlock += blockSize ;
      aBlock += blockSize ;
      oblock += blockSize ;
    }
    plane += planeStride ;
  }
}
