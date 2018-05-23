// @file vl_nnfmadbnorm.cu
// @brief Batch normalization MEX wrapper
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Sebastien Ehrhardt and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include <bits/mexutils.h>
#include <bits/nnfmadbnorm.hpp>
#include <bits/datamex.hpp>

#if ENABLE_GPU
#include <bits/datacu.hpp>
#endif

#include <assert.h>

/* option codes */
enum {
  opt_verbose = 0,
  opt_epsilon,
  opt_moments,
} ;

/* options */
VLMXOption  options [] = {
  {"Verbose", 0, opt_verbose },
  {"Epsilon", 1, opt_epsilon },
  {0, 0, 0}
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
   Resetting the context here resolves a crash when MATLAB quits and
   the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_MULTIPLIERS, IN_DERDATA,
            IN_DERMULTIPLIERS, IN_DERBIASES, IN_END
} ;

enum {
  OUT_RESULT = 0,
  OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  double epsilon = 1e-4 ;

  int opt ;
  int verbosity = 0 ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 5) {
    mexErrMsgTxt("The arguments are less than five.") ;
  }
  if (nin > 5 && vlmxIsString(in[5],-1)) {
    next = 5 ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {

      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_epsilon :
        if (!vlmxIsPlainScalar(optarg)) {
          mexErrMsgTxt("EPSILON is not a plain scalar.") ;
        }
        epsilon = mxGetPr(optarg)[0] ;
        break ;

      default:
        break ;
    }
  }

  vl::MexTensor data(context) ;
  vl::MexTensor multipliers(context);
  vl::MexTensor derData(context) ;
  vl::MexTensor derMultipliers(context);
  vl::MexTensor derBiases(context);

  data.init(in[IN_DATA]) ;
  data.reshape(4) ;

  multipliers.init(in[IN_MULTIPLIERS]) ;
  multipliers.reshape(1) ;

  derData.init(in[IN_DERDATA]) ;
  derData.reshape(4) ;

  derMultipliers.init(in[IN_DERMULTIPLIERS]) ;
  derMultipliers.reshape(1) ;

  derBiases.init(in[IN_DERBIASES]) ;
  derBiases.reshape(1) ;

  /* Check for GPU/data class consistency */
  if (! vl::areCompatible(data, multipliers)) {
    mexErrMsgTxt("DATA and MULTIPLIERS do not have compatible formats.") ;
  }
  if (! vl::areCompatible(derData, derMultipliers)) {
    mexErrMsgTxt("DERDATA and DERMULTIPLIERS do not have compatible formats.") ;
  }
  if (! vl::areCompatible(derData, derBiases)) {
    mexErrMsgTxt("DERDATA and DERBIASES do not have compatible formats.") ;
  }
  if (! vl::areCompatible(data, derData)) {
    mexErrMsgTxt("DATA and DERDATA do not have compatible formats.") ;
  }
  if (data.getShape() != derData.getShape()) {
    mexErrMsgTxt("DATA and DERDATA do not have the same size.") ;
  }

  /* Get the filter geometry */
  vl::TensorShape multipliersGeom(multipliers) ;
  if (multipliersGeom.getHeight() != data.getDepth()) {
    mexErrMsgTxt("The MULTIPLIERS size does not match the DATA depth.") ;
  }
  vl::TensorShape derMultipliersGeom(derMultipliers) ;
  if (derMultipliersGeom.getHeight() != derData.getDepth()) {
    mexErrMsgTxt("The DERMULTIPLIERS size does not match the DERDATA depth.") ;
  }
  vl::TensorShape derBiasesGeom(derBiases) ;
  if (derBiasesGeom.getHeight() != derData.getDepth()) {
    mexErrMsgTxt("The DERBIASES size does not match the DERDATA depth.") ;
  }

  /* Create output buffers */
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::MexTensor output(context) ;

  output.init(deviceType, dataType, data.getShape()) ;

  if (verbosity > 0) {
    mexPrintf("vl_nnfmadbnorm: mode %s; %s; moments %s/%s\n",
              (data.getDeviceType()==vl::VLDT_GPU)?"gpu":"cpu") ;
    vl::print("vl_nnfmadbnorm: data: ", data) ;
    vl::print("vl_nnfmadbnorm: multipliers: ", multipliers) ;
    vl::print("vl_nnfmadbnorm: derData: ", derData) ;
    vl::print("vl_nnfmadbnorm: derMultipliers: ", derMultipliers) ;
    vl::print("vl_nnfmadbnorm: derBiases: ", derBiases) ;
    mexPrintf("vl_nnfmadbnorm: epsilon: %f\n", epsilon) ;
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;
  vl::nn::FmadBatchNorm op(context,epsilon) ;

  error = op.forward(output, data, multipliers,
                     derData, derMultipliers, derBiases) ;

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  out[OUT_RESULT] = output.relinquish() ;
}
