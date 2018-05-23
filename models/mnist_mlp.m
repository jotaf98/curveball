function prediction = mnist_mlp(varargin)
%MNIST_MLP Returns a sigmoid MLP model for MNIST

  opts.init = 1;  % standard deviation for parameter initialization, or 'xavier' method
  opts.normalizeInput = true;  % normalize to range 0-1, or leave at 0-255
  opts.finalActivation = false;  % add activation after prediction, according to the KFAC paper
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;


  tanh = Layer.fromFunction(@vl_nntanh);  % new vl_nntanh function is not overloaded by AutoNN Layer


  images = Input('name', 'images', 'gpu', true) ;  % default input layer

  if opts.normalizeInput
    images = images / 255;
    images.precious = true;
  end

  x = vl_nnconv(images, 'size', [28, 28, 1, 128], 'weightScale', opts.init) ;
  x = tanh(x) ;

  x = vl_nnconv(x, 'size', [1, 1, 128, 64], 'weightScale', opts.init) ;
  x = tanh(x) ;

  x = vl_nnconv(x, 'size', [1, 1, 64, 32], 'weightScale', opts.init) ;
  x = tanh(x) ;

  prediction = vl_nnconv(x, 'size', [1, 1, 32, 10], 'weightScale', opts.init) ;

  if opts.finalActivation
    prediction = tanh(prediction) ;
  end

  % default training options for this network
  defaults.numEpochs = 50 ;
  defaults.batchSize = 64 ;
  defaults.learningRate = 0.01 ;
  defaults.weightDecay = 0 ;
  defaults.imageSize = [28, 28, 1] ;
  prediction.meta = defaults ;

end
