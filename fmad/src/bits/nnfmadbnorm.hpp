// @file nnfmadbnorm.hpp
// @brief Batch normalizatoion block
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
 Copyright (C) 2015-16 Sebastien Ehrhardt and Andrea Vedaldi.
 All rights reserved.

 This file is part of the VLFeat library and is made available under
 the terms of the BSD license (see the COPYING file).
 */

#ifndef __vl__nnfmadbnorm__
#define __vl__nnfmadbnorm__

#include <bits/data.hpp>
#include <stdio.h>

namespace vl { namespace nn {

  class FmadBatchNorm {
  public:
    FmadBatchNorm(vl::Context &context, double epsilon) ;

    vl::ErrorCode forward(vl::Tensor &output,
                          vl::Tensor const &input,
                          vl::Tensor const &multiplier,
                          vl::Tensor const &derInput,
                          vl::Tensor const &derMultiplier,
                          vl::Tensor const &derBias) ;

    vl::Context& context ;
    double epsilon ;
  } ;

} }

#endif /* defined(__vl__nnfmadbnorm__) */
