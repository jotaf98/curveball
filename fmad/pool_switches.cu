
// NOTE: the meanings of x/y here are switched.
// Code assumes dimensions are x, y, channels, samples.

__global__ void pool_switches 
(unsigned int* idx,
 const float* data,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledVolume,
 const int width,
 const int height,
 const int poolWidth,
 const int poolHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {
    int px = pooledIndex ;
    int py = px / pooledWidth ;
    int pz = py / pooledHeight ;
    px %= pooledWidth ;
    py %= pooledHeight ;
    data += pz * (width*height) ;  // offset by channel/sample

    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = min(x1 + poolWidth, width) ;
    int y2 = min(y1 + poolHeight, height) ;
    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;

    unsigned int bestIdx = y1 * width + x1 ;
    float value, bestValue = data[bestIdx] ;

    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        value = data[y * width + x] ;
        if (value > bestValue) {
          bestValue = value ;
          bestIdx = y * width + x ;
        }
      }
    }
    // return best index. must add the channel/sample offset, plus 1 for one-based indexes
    idx[pooledIndex] = bestIdx + pz * (width*height) + 1 ;
  }
}
