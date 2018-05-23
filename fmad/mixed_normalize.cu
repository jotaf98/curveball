__global__ void mixed_normalize(float * moments,
							    float* mu,
                                unsigned int numChannels,
                                float mass)
{
  int unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < numChannels){
    // max(0, __) is for numerical issues
    float mu_ = mu[i] ;
    float mean = moments[i] / mass ;
    float sigma2 = moments[i + numChannels]/mass - mean*mu_ ;
    moments[i] = mean ;
    moments[i + numChannels] = sigma2 ;
  }
}
