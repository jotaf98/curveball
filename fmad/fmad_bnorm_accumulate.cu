// Stolen from Seb

/*#include <cstdint>*/
#include <stdint.h>

#define WARP_SIZE 32

// macro function
#define min(a,b) (a > b ? b : a);

// -------------------------------------------------------------------
// helper functions
// -------------------------------------------------------------------

// Get largest memory address that is aligned to a warp worth of floats
// and smaller than x.
__forceinline__ __device__ uintptr_t getBlockBeginning(void const * x)
{
  return (uintptr_t)(x) & (~((uintptr_t)(WARP_SIZE*sizeof(float)) - 1)) ;
}

__forceinline__ __device__ void blockReduce2(volatile float * mdata,
                                             volatile float * sdata,
                                             unsigned int tid,
                                             unsigned int blockSize,
                                             unsigned int maxDataSize)
{
  __syncthreads();
  if (blockSize >= 1024 && maxDataSize + WARP_SIZE >=512) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; mdata[tid] += mdata[tid + 512]; } __syncthreads(); }
  if (blockSize >= 512  && maxDataSize + WARP_SIZE >=256) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; mdata[tid] += mdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256  && maxDataSize + WARP_SIZE >=128) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; mdata[tid] += mdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128  && maxDataSize + WARP_SIZE >=64)  { if (tid <  64) { sdata[tid] += sdata[tid + 64];  mdata[tid] += mdata[tid + 64];  } __syncthreads(); }
  if (tid < 32) {
    if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; mdata[tid] += mdata[tid + 32]; }
    if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; mdata[tid] += mdata[tid + 16]; }
    if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; mdata[tid] += mdata[tid +  8]; }
    if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; mdata[tid] += mdata[tid +  4]; }
    if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; mdata[tid] += mdata[tid +  2]; }
    if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; mdata[tid] += mdata[tid +  1]; }
  }
}



// This kernel accumulates means and variances for the data.
// Each block of thread sums over one or more data planes, resulting
// in an array accumulator[] of dimension numBlocksPerChannel x 2*numChannels.
//
// If each thread block scans all the images, then numBlocksPerChannel = 1.
// However, for efficiency different thread blocks do different
// subset of images, resulting in numBlocksPerChannel partial results to be summed
// later by a second kernel.
//
// The first part accumulator[:,0:numChannels-1] stores the data for the mean
// and the second part accumulator[:,numChannels,2*numChannels-1] the data
// for the sigmas.
//
// This function uses the sliding-window summing technique described
// above. It requires
//
//    2*sizeof(float)*blockSize
//
// bytes of shared scratch memory to hold to hold partial sums for
// means and sigmas.
__global__ void fmad_bnorm_accumulate(float * accumulator,
                                      float const * xData,
                                      float const * dxData,
                                      int planeArea,
                                      int numPlanes,
                                      int numChannels,
                                      int numBlocksPerChannel)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;

  extern __shared__ float s [] ;
  /*SharedMemory<float> smem ;*/
  /*float* s = smem.getPointer() ;*/
  float* mdata = s ;
  float* sdata = mdata + blockSize ;

  mdata[tid] = 0 ;
  sdata[tid] = 0 ;

  while (plane < numPlanes) {
    float const* xPlaneBegin = xData + plane * planeArea ;
    float const* xPlaneEnd = xPlaneBegin + planeArea ;
    float const* xBlock = (float const*) getBlockBeginning(xPlaneBegin) + tid ;

    float const* dxPlaneBegin = dxData + plane * planeArea ;
    float const* dxBlock = (float const*) getBlockBeginning(dxPlaneBegin) + tid ;

    while (xBlock < xPlaneEnd) {
      if (xBlock >= xPlaneBegin) {
        float x = *xBlock ;
        float dx = *dxBlock ;
        mdata[tid] += dx ;
        sdata[tid] += x * dx ;
      }
      xBlock += blockSize ;
      dxBlock += blockSize ;
    }
    plane += planeStride ;
  }

  blockReduce2(sdata, mdata, tid, blockSize, planeArea) ;

  if (tid == 0) {
    int chunk = blockIdx.x / numChannels ;
    int i = chunk + channel * numBlocksPerChannel ;
    accumulator[i] = mdata[0];
    accumulator[i + gridDim.x] = sdata[0];
  }
}

