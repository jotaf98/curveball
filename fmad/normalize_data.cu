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

__global__ void normalize_data(float * data,
                               float const * moments,
                               int planeArea,
                               int numPlanes,
                               int numChannels)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;

  float mean = moments[channel];
  float sigma = moments[channel+numChannels];
  /*float multiplier = multipliers[channel];*/
  /*float bias = biases[channel];*/
  /*float coefficient = 1.0 multiplier / sigma ;*/
  float coefficient = 1.0 / sigma ;

  while (plane < numPlanes) {
    float const * planeBegin = data + plane * planeArea ;
    float const * planeEnd = planeBegin + planeArea ;
    float const * block = (float const*) getBlockBeginning(planeBegin) + tid ;
    float * oblock = data + (block - data) ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        /**oblock = coefficient * (*block - mean) + bias ;*/
        *oblock = coefficient * (*block - mean) ;
      }
      block += blockSize ;
      oblock += blockSize ;
    }
    plane += planeStride ;
  }
}
