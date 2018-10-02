function idx = pool_switches(x, pool, stride, pad)
% POOL_SWITCHES
%   Max pooling switches/indexes.
%    idx = pool_switches(x, pool, stride, pad);
%    y = x(idx);
%   is the same as
%    y = vl_nnpool(x, pool, 'stride', stride, 'pad', pad);
%
%   Joao F. Henriques, 2017

  persistent kernel last_grid_size
  
  % release kernel
  if nargin == 1 && isequal(x, 'clear')
    clear kernel gridSize;
    return
  end

  % validate inputs
  assert(isa(x, 'gpuArray'), 'Only GPU mode is supported.');
  assert(strcmp(classUnderlying(x), 'single'), 'Only tensors of type single are supported.');

  if isscalar(pool), pool = pool([1 1]); end
  if isscalar(stride), stride = stride([1 1]); end
  if isscalar(pad)
    pad = pad([1 1 1 1]);
  else
    assert(numel(pad) == 4);  % [top bottom left right]
  end
  
  % compute pooled output size
  pooled_h = floor((size(x,1) - pool(1) + pad(1) + pad(2)) / stride(1)) + 1;
  pooled_w = floor((size(x,2) - pool(2) + pad(3) + pad(4)) / stride(2)) + 1;
  pooled_sz = [pooled_h, pooled_w, size(x,3), size(x,4)];
  
  % allocate output array
  idx = zeros(pooled_sz, 'uint32', 'gpuArray');
  
  MAX_NUM_THREADS = 1024;
  grid_size = ceil(prod(pooled_sz) / MAX_NUM_THREADS);

  if isempty(kernel) || ~existsOnGPU(kernel)
    % load the kernel if needed
    if ispc()
      ext = 'ptxw64';
    else
      ext = 'ptxa64';
    end
    kernel = parallel.gpu.CUDAKernel(['pool_switches.' ext], 'pool_switches.cu');
    kernel.ThreadBlockSize = MAX_NUM_THREADS;
    kernel.GridSize = grid_size;
    last_grid_size = grid_size;
    
  elseif grid_size ~= last_grid_size
    % kernel exists, but has different grid size
    kernel.GridSize = grid_size;
    last_grid_size = grid_size;
  end
  
  % launch the kernel
	idx = kernel.feval(idx, x, pooled_h, pooled_w, prod(pooled_sz), ...
    size(x,1), size(x,2), pool(1), pool(2), stride(1), stride(2), pad(1), pad(3));
end
