// @file nnbnorm.cu
// @brief Batch normalization block
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
 Copyright (C) 2015-17 Sebastien Ehrhardt and Andrea Vedaldi.
 All rights reserved.

 This file is part of the VLFeat library and is made available under
 the terms of the BSD license (see the COPYING file).
 */

#include <bits/nnfmadbnorm.hpp>
#include <bits/impl/dispatcher.hpp>
#include <cassert>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <iostream>

using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

template<DeviceType deviceType, DataType dataType> struct FmadBatchNormForward ;

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType>
struct FmadBatchNormForward<VLDT_CPU, dataType>
{
  vl::ErrorCode operator()(FmadBatchNorm &op,
                           Tensor &output,
                           Tensor const &input,
                           Tensor const &multiplier,
                           Tensor const &derInput,
                           Tensor const &derMultiplier,
                           Tensor const &derBias)
  {
    throw std::invalid_argument( "CPU mode is not (yet) supported for fmad bnorm") ;
  }
} ;

// -------------------------------------------------------------------
//                                                              Driver
// -------------------------------------------------------------------

#if ENABLE_GPU
#include <bits/nnfmadbnorm_gpu.cu>
#endif

FmadBatchNorm::FmadBatchNorm(Context &context, double epsilon):
context(context), epsilon(epsilon) { }

vl::ErrorCode
FmadBatchNorm::forward(Tensor &output,
                       Tensor const &input,
                       Tensor const &multiplier,
                       Tensor const &derInput,
                       Tensor const &derMultiplier,
                       Tensor const &derBias)
{
  return dispatch<FmadBatchNormForward>()(*this, output, input, multiplier,
                                            derInput, derMultiplier, derBias) ;
}
