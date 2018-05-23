function test_fwd_mode_autodiff_cnn()
  addpath ..

  gpu = 2;
  input_size = [10, 10, 1, 3];
  pool = 'max';  % max or avg
  weight_scale = 1;


  % a network similar to LeNet but much smaller for speed
  rng(0);

  images = Input('name', 'images', 'gpu', true);
  extraInput = Input('name', 'extra', 'gpu', true);
  extra_input_size = [1, 1, 10, 1];

  x = vl_nnconv(images, 'size', [3, 3, 1, 10], 'weightScale', weight_scale) ;
  x = vl_nnpool(x, 2, 'stride', 2, 'pad', 1, 'method', pool) ;
  x = vl_nndropout(x);
  x = vl_nnconv(x, 'size', [3, 3, 10, 15], 'weightScale', weight_scale) ;
  x = vl_nnmaxout(x, 5) ;
  x = vl_nnbnorm(x, Param('value',ones(1,3,'single')), ...
                    Param('value',zeros(1,3,'single')), ...
                    'moments', [ones(1,3,'single'),zeros(1,3,'single')]) ;
  x = vl_nnrelu(x) ;
  prediction = vl_nnconv(x, 'size', [3, 3, 3, 10], 'weightScale', weight_scale) ;

  % test derivatives for bsxfun times op
  prediction = prediction .* single(ones([1 1 1 3]) * 0.1) ;

  % test derivatives for bsxfun power op
  prediction = prediction .^ 2 ;

  % test derivatives for cat op on constants and Inputs
  prediction = cat(4, prediction, ones(1, 1, 10, 1, 'single')) ;
  prediction = cat(4, prediction, extraInput) ;

  % test derivatives for slcing op
  prediction = prediction([1 1 1 1]) ;


  % compile network
  net = Net(prediction);

  % select GPU
  net.useGpu(gpu);
  props = {'single'};
  if ~isempty(gpu)
    props{2} = 'gpuArray';
  end

  % create random output derivative to project to
  output_size = prediction.evalOutputSize('images', input_size, ...
                                          'extra', extra_input_size);
  prediction_der = randn(output_size, props{:});

  % assign input value (will be reused by all forward/backward calls)
  net.setValue(images, rand(input_size, props{:}));

  % assign extra input (used to test cat function)
  net.setValue(extraInput, rand(extra_input_size, props{:}));


  %
  % run reverse-mode auto-diff (RMAD, standard backprop, for validation)
  %

  forward(net);
  true_param_der = backward(net, prediction_der);


  %
  % run forward-mode auto-diff (FMAD), assumes forward() was ran before
  %

  params = [net.params.var];  % parameter var indexes
  input_der = cell(size(params));  % parameter derivatives, input to FMAD
  result_der = cell(size(params));  % will hold results, per-parameter

  for p = 1:numel(params)
    % set parameter forward-mode derivatives in the network to 0
    val = zeros(size(net.vars{params(p)}), props{:});
    input_der{p} = val;

    % allocate result tensor with the same size
    result_der{p} = val;
  end

  %profile on ;

  disp('Maximum error per parameter:');

  % iterate parameter tensors
  for p = 1:numel(params)
    for i = 1:numel(input_der{p})  % iterate elements
      % set one-hot tensor
      input_der{p}(i) = 1;

      % run FMAD to obtain the resulting tensor derivative
      tensor_der = fmad(net, input_der);

      % project derivative, and store the resulting scalar
      result_der{p}(i) = tensor_der(:)' * prediction_der(:);

      % undo one-hot tensor
      input_der{p}(i) = 0;
    end
    
    
    % validate
    disp(net.params(p).name);
    disp(max(abs(result_der{p}(:) - true_param_der{p}(:))));

%     figure(p), imagesc(result_der{p})
%     title(sprintf('FMAD der #%i', p))
%     figure(numel(params) + p), imagesc(true_param_der{p})
%     title(sprintf('RMAD (backprop) #%i', p))
  end

  %profile viewer

end

