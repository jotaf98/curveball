// After accumulation, we need to renormalize the moments.
//
// 1. It shoudl be called with enough threads to cover all
//    numChannels in the moments.
//
// 2. The actual number of blocks is determined based on the block
//    size to satisfy condition (2).

__global__ void normalize_moments(float * moments,
                                  unsigned int numChannels,
                                  float mass,
                                  float epsilon)
{
  int unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < numChannels){
    // max(0, __) is for numerical issues
    float mean = moments[i] / mass ;
    float sigma2 = max((float).0, moments[i + numChannels]/mass - mean*mean) ;
    moments[i] = mean ;
    moments[i + numChannels] = sqrt(sigma2 + epsilon);
  }
}
