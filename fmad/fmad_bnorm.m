function der = fmad_bnorm(x, g, b, dx, dg, db, varargin)
% FMAD_BNORM - compute forward derivatives for batch norm
%   DER = FMAD_BNORM(X, G, B, DX, DG, DB) computes forward mode derivatives
%   for the batch normalization operator for input X, gains G, biases B and
%   their coresponding forward projected forward derivatives DX, DG and DB.
%
%   The Newtonian Society, 2018

  opts.epsilon = 1e-4 ;
  opts.refreshKernels = false ;
  opts.MAX_NUM_THREADS = 1024 ;
  opts.debug = false ;
  opts = vl_argparse(opts, varargin) ;


  FLOAT_SZ = 4 ;

  % validate inputs
  assert(isa(x, 'gpuArray')) ;
  assert(isa(b, 'gpuArray')) ;
  assert(isa(g, 'gpuArray')) ;
  assert(isa(dx, 'gpuArray')) ;
  assert(isa(dg, 'gpuArray')) ;
  assert(isa(db, 'gpuArray')) ;

  assert(strcmp(classUnderlying(x), 'single')) ;
  assert(strcmp(classUnderlying(b), 'single')) ;
  assert(strcmp(classUnderlying(g), 'single')) ;
  assert(strcmp(classUnderlying(dx), 'single')) ;
  assert(strcmp(classUnderlying(dg), 'single')) ;
  assert(strcmp(classUnderlying(db), 'single')) ;

  % common values
  WARP_SIZE = 32 ;
  [H,W,C,N] = size(x) ;
  planeArea = H * W ;
  numPlanes = C * N ;
  mass = H * W * N ; % i.e. total number of channels

  % --------------------------------------------------------------------
  %                                                  MOMENT ACCUMULATION
  % --------------------------------------------------------------------

  persistent momentKernel ;

  % configure kernel
  grid_size = C ;
  numBlocksPerChannel = 1 ;
  rawMoments = zeros(C, 2, 'like', x) ;
  blockSize = getBlockSize(planeArea, opts.MAX_NUM_THREADS, WARP_SIZE) ;

  if isempty(momentKernel) || ~existsOnGPU(momentKernel) || opts.refreshKernels
    momentKernel = fetchKernel('accumulate_moments_partial') ;
    momentKernel.GridSize = grid_size ;
    momentKernel.ThreadBlockSize = blockSize ;
    momentKernel.SharedMemorySize = 2 * blockSize * FLOAT_SZ ;
  end

  rawMoments = momentKernel.feval(...
      rawMoments, x, planeArea, numPlanes, C, numBlocksPerChannel) ;

  % ---------------------------------------------------------------------
  %                                                  MOMENT NORMALIZATION
  % ---------------------------------------------------------------------

  persistent normalizeMomentKernel ;

  if isempty(normalizeMomentKernel) || ~existsOnGPU(normalizeMomentKernel) || opts.refreshKernels
    normalizeMomentKernel = fetchKernel('normalize_moments') ;

    % configure kernel
    grid_size = ceil(C / blockSize) ;
    normalizeMomentKernel.GridSize = grid_size ;
    normalizeMomentKernel.ThreadBlockSize = blockSize ;
  end

  % launch the moment kernels
  moments = normalizeMomentKernel.feval(rawMoments, C, mass, opts.epsilon) ;

  if opts.debug
    expectedMu = squeeze(mean(mean(mean(x, 1), 2), 4)) ;
    expectedMuReshaped = reshape(expectedMu, 1, 1, C, 1) ;
    expectedSig = squeeze(sum(sum(sum((x - ...
                           expectedMuReshaped).^2, 1), 2), 4)) ./ mass;
    rel_error = norm(abs(expectedSig - moments(:,2).^2)) ...
                                          / norm(expectedSig(:)) ;
    assert(rel_error < 1e-5, 'uh oh') ;
  end

  % ---------------------------------------------------------------------
  %                                                   INPUT NORMALIZATION
  % ---------------------------------------------------------------------

  persistent normalizeDataKernel ;

  if isempty(normalizeDataKernel) || ~existsOnGPU(normalizeDataKernel) || opts.refreshKernels
    normalizeDataKernel = fetchKernel('normalize_data') ;

    % configure kernel
    grid_size = C * 1 ;
    normalizeDataKernel.GridSize = grid_size ;
    normalizeDataKernel.ThreadBlockSize = blockSize ;
  end


  % launch the moment kernels
  xHat = normalizeDataKernel.feval(x, moments, planeArea, numPlanes, C) ;

  if opts.debug
    expected_xHat = vl_nnbnorm(x, gpuArray(ones(size(g), 'single')), ...
                                      gpuArray(zeros(size(b), 'single'))) ;
    rel_error = norm(abs(gather(expected_xHat(:)) - gather(xHat(:)))) ...
                           / norm(gather(xHat(:))) ;
    assert(rel_error < 1e-5, 'uh oh') ;
  end

  % ---------------------------------------------------------------------
  %                                                       FMAD ACCUMULATE
  % ---------------------------------------------------------------------
  % computing an aggregated version of (x * dx) - mu * dx

  persistent fmadAccumKernel ;

  if isempty(fmadAccumKernel) || ~existsOnGPU(fmadAccumKernel) || opts.refreshKernels
    fmadAccumKernel = fetchKernel('fmad_bnorm_accumulate') ;

    % configure kernel
    grid_size = C * 1 ;
    fmadAccumKernel.ThreadBlockSize = blockSize ;
    fmadAccumKernel.GridSize = grid_size ;
    fmadAccumKernel.SharedMemorySize = 2 * blockSize * FLOAT_SZ ;
  end

  accumMoments = zeros(C, 2, 'like', x) ;


  % launch the moment kernels
  accumMoments = fmadAccumKernel.feval(...
      accumMoments, x, dx, planeArea, numPlanes, C, numBlocksPerChannel) ;


  if opts.debug
    dx_mu = squeeze(gather(sum(sum(sum(dx, 1), 2), 4))) ;
    rel_error = norm(abs(dx_mu - accumMoments(:,1))) / norm(dx_mu(:)) ;
    assert(rel_error < 1e-5, 'uh oh') ;

    tmp = squeeze(gather(sum(sum(sum(x .* dx, 1), 2), 4))) ;
    rel_error = norm(abs(tmp - accumMoments(:,2))) / norm(tmp(:)) ;
    assert(rel_error < 1e-5, 'uh oh') ;
  end

  % ---------------------------------------------------------------------
  %                                                  MOMENT NORMALIZATION
  % ---------------------------------------------------------------------

  persistent mixedMomentKernel ;

  if isempty(mixedMomentKernel) || ~existsOnGPU(mixedMomentKernel) || opts.refreshKernels
    mixedMomentKernel = fetchKernel('mixed_normalize') ;

    % configure kernel
    grid_size = ceil(C / blockSize) ;
    mixedMomentKernel.GridSize = grid_size ;
    mixedMomentKernel.ThreadBlockSize = blockSize ; % MAX_NUM_THREADS ;
  end


  % launch the moment kernels
  mixedMoments = mixedMomentKernel.feval(accumMoments, moments, C, mass) ;
  %disp('inside fmad_bnorm.m') ;
  %der = mixedMoments ; return

  if opts.debug
    mtimes = x .* dx ;
    x_hat_der_sig = sum(sum(sum(mtimes, 1), 2), 4) ./ mass ;
    offset = (dx_mu / mass) .* moments(:,1) ;
    x_hat_der_sig = x_hat_der_sig - reshape(offset, 1, 1, []) ;
    x_hat_der_sig = gather(squeeze(x_hat_der_sig)) ;
    expectedMu = squeeze(mean(mean(mean(x, 1), 2), 4)) ;
    expectedMuReshaped = reshape(expectedMu, 1, 1, C, 1) ;
    expectedSig = squeeze(sum(sum(sum((x - ...
                           expectedMuReshaped).^2, 1), 2), 4)) ./ mass;
    rel_error = norm(abs(expectedSig - moments(:,2).^2)) ...
                                          / norm(expectedSig(:)) ;
    assert(rel_error < 1e-5, 'uh oh') ;
  end

  % ---------------------------------------------------------------------
  %                                                      DX NORMALIZATION
  % ---------------------------------------------------------------------

  % configure updated kernel (may need to fix depending on benchmarks)
  grid_size = C * 1 ;
  normalizeDataKernel.GridSize = grid_size ;
  normalizeDataKernel.ThreadBlockSize = blockSize ; % MAX_NUM_THREADS ;

  % launch the moment kernels
  dxMoments = [mixedMoments(:,1) moments(:,2)] ;
  dxHatIsh = normalizeDataKernel.feval(dx, dxMoments, planeArea, numPlanes, C) ;
  %disp('inside fmad_bnorm.m') ;
  %der = dxHatIsh ; return

  if opts.debug
    expected_xHat = vl_nnbnorm(dx, gpuArray(ones(size(g), 'single')), ...
                                      gpuArray(zeros(size(b), 'single')), ...
                                'Moments', dxMoments) ;
    rel_error = norm(abs(gather(expected_xHat(:)) - gather(dxHatIsh(:)))) ...
                           / norm(gather(dxHatIsh(:))) ;
    assert(rel_error < 1e-5, 'uh oh') ;
  end

  % ---------------------------------------------------------------------
  %                                                           MEGA KERNEL
  % ---------------------------------------------------------------------
  % The mega kernel performs the following computations
  %  beta = xHat .* xHat_der_sig ./ (sig_sq + epsilon) ;*/
  %  dxHat = alpha - beta ;*/
  %  der = g .* dx_hat + xHat .* dg + db ;*/

  persistent megaKernel ;

  if isempty(megaKernel) || ~existsOnGPU(megaKernel) || opts.refreshKernels
    megaKernel = fetchKernel('mega_kernel') ;

    % configure kernel
    grid_size = C * 1 ;
    megaKernel.GridSize = grid_size ;
    megaKernel.ThreadBlockSize = blockSize ; % MAX_NUM_THREADS ;
  end


	alpha = dxHatIsh ;
  xHatDerSig = mixedMoments(:,2) ;
  %xHatDerSig = 0 * ones(size(xHatDerSig), 'like', xHatDerSig) ;
  %moments = ones(size(moments), 'like', moments) ;
  %alpha = ones(size(alpha), 'like', alpha) ;

  % launch the moment kernels
  der = megaKernel.feval(xHat, alpha, moments, xHatDerSig, ...
                         g, dg, db, opts.epsilon, planeArea, numPlanes, C) ;

  if opts.debug
    sig_sq = reshape(moments(:,2) .^ 2, 1, 1, []) ;
    beta = xHat .* reshape(x_hat_der_sig,1,1,[]) ./ (sig_sq + opts.epsilon) ;
    dx_hat = alpha - beta ;
    der2 = reshape(g,1,1,[]) .* dx_hat ...
                      + xHat .* reshape(dg,1,1,[]) + reshape(db,1,1,[]) ;
    rel_error = norm(gather(abs(der(:) - der2(:)))) / norm(gather(der(:))) ;
    assert(rel_error < 1e-5, 'uh oh') ;
  end
end

% -------------------------------------------------------------------
function kernel = fetchKernel(kernelName)
% -------------------------------------------------------------------
  if ispc(), ext = 'ptxw64' ; else, ext = 'ptxa64' ; end
  kernelDir = fullfile(newton_root, 'matlab') ;
  compiled = fullfile(kernelDir, sprintf(['%s.' ext], kernelName)) ;
  src = fullfile(kernelDir, sprintf('%s.cu', kernelName)) ;
  kernel = parallel.gpu.CUDAKernel(compiled, src) ;
end

% --------------------------------------------------------------------
function blockSize = getBlockSize(dataSize, MAX_NUM_THREADS, WARP_SIZE)
% --------------------------------------------------------------------
  blockSize = MAX_NUM_THREADS / 2 ;
  if dataSize < blockSize
    numWarps = dataSize / WARP_SIZE ;

    if numWarps < 4
      blockSize = 2 * WARP_SIZE ;
    elseif numWarps < 8
      blockSize = 4 * WARP_SIZE ;
    else
      blockSize = 8 * WARP_SIZE ;
    end

  end
  return
end
