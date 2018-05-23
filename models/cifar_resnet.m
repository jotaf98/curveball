function prediction = cifar_resnet(varargin)
%CIFAR_RESNET Returns a ResNet model for CIFAR

  opts.dropout = 0;  % optional dropout rate
  [opts, varargin] = vl_argparse(opts, varargin, 'nonrecursive') ;
  

  % any extra arguments are passed to the ResNet constructor
  prediction = models.ResNet('numClasses',10, 'variant','18', varargin{:});
  assert(isequal(prediction.func, @vl_nnconv));  % sanity check
  
  % the last few layers are: relu -> mean -> mean -> conv.
  % skip the pooling (2 mean operations) by connecting the relu directly to
  % the conv.
  relu = prediction.find(@vl_nnrelu, -1);
  prediction.inputs{1} = relu;
  
  
  if opts.dropout ~= 0
    % add dropout layers.
    % first, find sums (end of residual block)
    sums = prediction.find(@vl_nnwsum);
    for i = 1:numel(sums)
      % get input bnorm
      bnorm = sums{i}.inputs{2};
      assert(isequal(bnorm.func, @vl_nnbnorm_wrapper));
      
      % get input conv
      conv = bnorm.inputs{1};
      assert(isequal(conv.func, @vl_nnconv));
      
      % insert dropout between conv and its input
      conv.inputs{1} = vl_nndropout(conv.inputs{1}, 'rate', opts.dropout);
    end
    
  end
  
  
  % update meta-information for CIFAR
  meta.imageSize = [32, 32, 3];
  meta.learningRate = [0.01 * ones(1,3) 0.1 * ones(1,80) 0.01 * ones(1,10) 0.001 * ones(1,27)];
  meta.numEpochs = numel(meta.learningRate);
  meta.batchSize = 128;
  meta.momentum = 0.95;
  meta.weightDecay = 0;
  
  prediction.meta = meta;

end
