// Stolen from Seb

#include <bits/datacu.hpp>
#include <bits/nnfmadbnorm.hpp>
#include <bits/impl/blashelper.hpp>
#include <bits/impl/sharedmem.cuh>
#include <cassert>
#include <cstdint>
#include <cfloat>
#include <stdint.h>
#include <stdexcept>
#define WARP_SIZE 32

// macro function
#define min(a,b) (a > b ? b : a);

// -------------------------------------------------------------------
// helper functions
// -------------------------------------------------------------------

// get the smallest x which is a multiple of factor
static inline int nextMultipleOf(int x, int factor)
{
  return factor * ((x + factor - 1)/factor) ;
}

static inline int getBlockSize(int dataSize)
{
  int blockSize = VL_CUDA_NUM_THREADS / 2 ;
  if (dataSize < blockSize) {
    unsigned int numWarps = dataSize / WARP_SIZE ;
    if (numWarps < 4) {
      blockSize = 2 * WARP_SIZE ;
    }
    else if (numWarps < 8) {
      blockSize = 4 * WARP_SIZE ;
    }
    else {
      blockSize = 8 * WARP_SIZE ;
    }
  }
  return blockSize ;
}

// Get largest memory address that is aligned to a warp worth of floats
// and smaller than x.
__forceinline__ __device__ uintptr_t getBlockBeginning(void const * x)
{
  return (uintptr_t)(x) ; //& (~((uintptr_t)(WARP_SIZE*sizeof(float)) - 1)) ;
  /*return (uintptr_t)(x) & (~((uintptr_t)(WARP_SIZE*sizeof(float)) - 1)) ;*/
}


template<typename T>
__forceinline__ __device__ void blockReduce2(volatile T * mdata,
                                             volatile T * sdata,
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
template<typename T>
__global__ void accumulate_moments_partial(T * accumulator,
                                           T const * data,
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

  SharedMemory<T> smem ;
  T * s = smem.getPointer() ;
  T* mdata = s ;
  T* sdata = mdata + blockSize ;

  mdata[tid] = 0 ;
  sdata[tid] = 0 ;

  while (plane < numPlanes) {
    T const * planeBegin = data + plane * planeArea ;
    T const * planeEnd = planeBegin + planeArea ;
    T const * block = (T const*) getBlockBeginning(planeBegin) + tid ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        T x = *block ;
        mdata[tid] += x ;
        sdata[tid] += x * x ;
      }
      block += blockSize ;
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

// After accumulation, we need to renormalize the moments.
//
// 1. It shoudl be called with enough threads to cover all
//    numChannels in the moments.
//
// 2. The actual number of blocks is determined based on the block
//    size to satisfy condition (2).

template<typename T>
__global__ void normalize_moments(T * moments,
                                  unsigned int numChannels,
                                  T mass,
                                  T epsilon)
{
  int unsigned i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i < numChannels){
    // max(0, __) is for numerical issues
    T mean = moments[i] / mass ;
    T sigma2 = max((T).0, moments[i + numChannels]/mass - mean*mean) ;
    moments[i] = mean ;
    moments[i + numChannels] = sqrt(sigma2 + epsilon);
  }
}

template<typename T>
__global__ void normalize_data2(T * output,
							   T const * data,
                               T const * moments,
                               int planeArea,
                               int numPlanes,
                               int numChannels)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;

  T mean = moments[channel+2*numChannels];
  T sigma = moments[channel+numChannels];
  /*T multiplier = multipliers[channel];*/
  /*T bias = biases[channel];*/
  /*T coefficient = 1.0 multiplier / sigma ;*/
  T coefficient = 1.0 / sigma ;

  while (plane < numPlanes) {
    T const * planeBegin = data + plane * planeArea ;
    T const * planeEnd = planeBegin + planeArea ;
    T const * block = (T const*) getBlockBeginning(planeBegin) + tid ;

    T const * outputPlaneBegin = output + plane * planeArea ;
    T const * outputBlock = (T const*) getBlockBeginning(outputPlaneBegin) + tid ;
    T * oblock = output + (outputBlock - output) ;

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
// Call this kernel like compute_moments, but it does not need a scratch space

template<typename T>
__global__ void normalize_data(T * output,
							   T const * data,
                               T const * moments,
                               int planeArea,
                               int numPlanes,
                               int numChannels)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;

  T mean = moments[channel];
  T sigma = moments[channel+numChannels];
  /*T multiplier = multipliers[channel];*/
  /*T bias = biases[channel];*/
  /*T coefficient = 1.0 multiplier / sigma ;*/
  T coefficient = 1.0 / sigma ;

  while (plane < numPlanes) {
    T const * planeBegin = data + plane * planeArea ;
    T const * planeEnd = planeBegin + planeArea ;
    T const * block = (T const*) getBlockBeginning(planeBegin) + tid ;

    T const * outputPlaneBegin = output + plane * planeArea ;
    T const * outputBlock = (T const*) getBlockBeginning(outputPlaneBegin) + tid ;
    T * oblock = output + (outputBlock - output) ;

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
template<typename T>
__global__ void fmad_bnorm_accumulate(T * accumulator,
                                      T const * xData,
                                      T const * dxData,
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

  SharedMemory<T> smem ;
  T* s = smem.getPointer() ;
  T* mdata = s ;
  T* sdata = mdata + blockSize ;

  mdata[tid] = 0 ;
  sdata[tid] = 0 ;

  while (plane < numPlanes) {
    T const* xPlaneBegin = xData + plane * planeArea ;
    T const* xPlaneEnd = xPlaneBegin + planeArea ;
    T const* xBlock = (T const*) getBlockBeginning(xPlaneBegin) + tid ;

    T const* dxPlaneBegin = dxData + plane * planeArea ;
    T const* dxBlock = (T const*) getBlockBeginning(dxPlaneBegin) + tid ;

    while (xBlock < xPlaneEnd) {
      if (xBlock >= xPlaneBegin) {
        T x = *xBlock ;
        T dx = *dxBlock ;
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

// TODO(samuel): comment
template<typename T>
__global__ void mixed_normalize(T * moments,
							    T* mu,
                                unsigned int numChannels,
                                T mass)
{
  int unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < numChannels){
    // max(0, __) is for numerical issues
    T mu_ = mu[i] ;
    T mean = moments[i] / mass ;
    T sigma2 = moments[i + numChannels]/mass - mean*mu_ ;
    moments[i] = mean ;
    moments[i + numChannels] = sigma2 ;
  }
}

// Call this kernel like compute_moments, but it does not need a scratch space
// DOCS
/*      beta = x_hat .* x_hat_der_sig ./ (sig_sq + epsilon) ;*/
/*      dx_hat = alpha - beta ;*/
/*      der = g .* dx_hat + x_hat .* dg + db ;*/
/*
//      dx_hat = alpha - x_hat .* x_hat_der_sig ./ (sig_sq + epsilon) ;
//      der = g .* dx_hat + x_hat .* dg + db ;*/

template<typename T>
__global__ void mega_kernel(T* xHat,
                            T const * alpha,
                            T const * moments,
                            T const * xHatDerSig,
                            T const * gains,
                            T const * dGains,
                            T const * dBias,
                            T const epsilon,
                            int planeArea,
                            int numPlanes,
                            int numChannels)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;

  T sigma2 = moments[channel+numChannels] * moments[channel+numChannels] ;
  T x_hat_der_sig = xHatDerSig[channel] ;
  T g = gains[channel] ;
  T dg = dGains[channel] ;
  T db = dBias[channel] ;

  T coefficient =  x_hat_der_sig / (sigma2 + epsilon) ;
  T dxHat ;

  while (plane < numPlanes) {
    T const * planeBegin = xHat + plane * planeArea ;
    T const * planeEnd = planeBegin + planeArea ;
    T const * xHatBlock = (T const*) getBlockBeginning(planeBegin) + tid ;

    T const * aPlaneBegin = alpha + plane * planeArea ;
    T const * aBlock = (T const*) getBlockBeginning(aPlaneBegin) + tid ;

    T * oblock = xHat + (xHatBlock - xHat) ;

    while (xHatBlock < planeEnd) {
      if (xHatBlock >= planeBegin) {
        /*dxHat = *aBlock - coefficient * (*xHatBlock) ;*/
        T ab = *aBlock;
        dxHat = ab - coefficient * (*xHatBlock) ;
        T interim = *xHatBlock ;
        *oblock = g * dxHat + dg * interim + db ;
      }
      xHatBlock += blockSize ;
      aBlock += blockSize ;
      oblock += blockSize ;
    }
    plane += planeStride ;
  }
}

template<DataType dataType>
struct FmadBatchNormForward<VLDT_GPU, dataType>
{
  vl::ErrorCode operator()(FmadBatchNorm &op,
                           Tensor &output,
                           Tensor const &input,
                           Tensor const &multiplier,
                           Tensor const &derInput,
                           Tensor const &derMultiplier,
                           Tensor const &derBias)
  {
    cudaError_t status ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto height = input.getHeight() ;
    auto width = input.getWidth() ;
    auto numChannels = input.getDepth() ;
    auto size = input.getSize() ;

    auto outputData = (type*)output.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto multiplierData = (type const*)multiplier.getMemory() ;

    size_t planeArea = height * width ;
    size_t numPlanes = numChannels * size ;

    // Compute number compute chunks.
    size_t blockSize = getBlockSize(planeArea) ;
    size_t numBlocksPerChannel = 1  ;
    size_t numBlocks = numChannels * numBlocksPerChannel ;

    // Get scratch space.
    size_t accumulatorSize = (numBlocksPerChannel == 1) ? 0 : 2 * nextMultipleOf(numBlocks, WARP_SIZE) ;

    size_t workspaceSize = accumulatorSize + 2 * numChannels ;
    type * workspace = (type*)op.context.getWorkspace(vl::VLDT_GPU, (2 * workspaceSize+numPlanes*planeArea)* sizeof(type)) ;
    if (workspace == NULL && workspaceSize > 0) {
      return VLE_OutOfMemory ;
    }
    type * accumulatorData = workspace ;


    // Accumulate moments.
    if (numBlocksPerChannel > 1) {
      throw std::invalid_argument( "Not yet supported") ;
    } else { // Total directly.
      accumulate_moments_partial<type><<<numBlocks, blockSize, 2*blockSize*sizeof(type)>>>
      (accumulatorData,
       inputData,
       planeArea,
       numPlanes,
       numChannels,
       1) ;
    }

    // Normalize moments.
    type mass = planeArea*size;
    normalize_moments<type><<<divideAndRoundUp(numChannels,blockSize),blockSize>>>
    (accumulatorData, numChannels, mass, (type)op.epsilon) ;


    // Normalize the data and apply multipliers and bias.
    normalize_data<type><<<numBlocks, blockSize>>>
    (outputData, inputData, accumulatorData, planeArea, numPlanes, numChannels) ;

    // ---------------------------------------------------------------------
    //                                                       FMAD ACCUMULATE
    // ---------------------------------------------------------------------
    // computing an aggregated version of (x * dx) - mu * dx

    auto derInputData = (type const*)derInput.getMemory() ;
    auto derMultiplierData = (type const*)derMultiplier.getMemory() ;
    auto derBiasData = (type const*)derBias.getMemory() ;

    type * accumMoments = workspace + workspaceSize;

    // Normalize the data and apply multipliers and bias.
    fmad_bnorm_accumulate<type><<<numBlocks, blockSize, 2*blockSize*sizeof(type)>>>
    (accumMoments, inputData, derInputData, planeArea, numPlanes, numChannels, numBlocksPerChannel) ;

	// ---------------------------------------------------------------------
	//                                                  MOMENT NORMALIZATION
	// ---------------------------------------------------------------------

    // Normalize the data and apply multipliers and bias.
    mixed_normalize<type><<<divideAndRoundUp(numChannels,blockSize),blockSize>>>
    (accumMoments, accumulatorData, numChannels, mass) ;

    // ---------------------------------------------------------------------
    //                                                      DX NORMALIZATION
    // ---------------------------------------------------------------------
    // Normalize the data and apply multipliers and bias.
    type * normalizedData = workspace + 2*workspaceSize;
    normalize_data2<type><<<numBlocks, blockSize>>>
    (normalizedData, derInputData, accumulatorData, planeArea, numPlanes, numChannels) ;

	// ---------------------------------------------------------------------
	//                                                           MEGA KERNEL
	// ---------------------------------------------------------------------
	// The mega kernel performs the following computations
	//  beta = xHat .* xHat_der_sig ./ (sig_sq + epsilon) ;*/
	//  dxHat = alpha - beta ;*/
	//  der = g .* dx_hat + xHat .* dg + db ;*/

    /*outputData = normalizedData ;*/
    // Apply the MEGA KERNEL.
    type * xHatDerSig = accumMoments + numChannels ;
    //cudaMemcpy(accumMoments, normalizedData, dataWorkspaceSize * sizeof(type), cudaMemcpyDeviceToDevice) ;
    mega_kernel<type><<<numBlocks,blockSize>>>
    (outputData, normalizedData, accumulatorData, xHatDerSig, multiplierData,
     derMultiplierData, derBiasData, (type)op.epsilon, planeArea, numPlanes,
     numChannels) ;

    status = cudaPeekAtLastError() ;
    return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
 }
} ;
