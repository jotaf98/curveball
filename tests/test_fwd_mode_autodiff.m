function test_fwd_mode_autodiff()
  addpath ..

  x_dim = 6;  % input size
  h_dim = 6;  % hidden layer size
  y_dim = 10;  % output size
  num_samples = 8;
  num_layers = 2;

  rng(0);

  % 1 or 2-layers linear network model, large initial magnitudes
  x = Input();
  w1 = Param('value', randn(h_dim, x_dim, 'single'));
  b1 = Param('value', randn(h_dim, 1, 'single'));
  w2 = Param('value', randn(y_dim, h_dim, 'single'));
  b2 = Param('value', randn(y_dim, 1, 'single'));

  x2 = w1 * x + b1;  % first layer


  if num_layers == 1
    prediction = x2;
    y_dim = h_dim;
  else
    x2 = vl_nnrelu(x2);  % activation?

    prediction = w2 * x2 + b2;  % second layer
  end

  Layer.workspaceNames();
  prediction.sequentialNames();


  % compile network
  net = Net(prediction);

  % assign input value (will be reused by all forward/backward calls)
  x_value = rand(x_dim, num_samples, 'single');
  net.setValue(x, x_value);

  % create random output derivative to project to
  y_der = randn(y_dim, num_samples, 'single');


  %
  % run reverse-mode auto-diff (RMAD, standard backprop, for validation)
  %

  forward(net);
  true_param_der = backward(net, y_der);


  %
  % run forward-mode auto-diff (FMAD), assumes forward() was ran before
  %

  params = [net.params.var];  % parameter var indexes
  input_der = cell(size(params));  % parameter derivatives, input to FMAD
  result_der = cell(size(params));  % will hold results, per-parameter

  for p = 1:numel(params)
    % set parameter forward-mode derivatives in the network to 0
    val = zeros(size(net.vars{params(p)}), 'single');
    input_der{p} = val;

    % allocate result tensor with the same size
    result_der{p} = val;
  end

  % iterate parameter tensors
  for p = 1:numel(params)
    for i = 1:numel(input_der{p})  % iterate elements
      % set one-hot tensor
      input_der{p}(i) = 1;

      % run FMAD to obtain the resulting tensor derivative
      tensor_der = fmad(net, input_der);

      % project derivative, and store the resulting scalar
      result_der{p}(i) = tensor_der(:)' * y_der(:);

      % undo one-hot tensor
      input_der{p}(i) = 0;
    end
  end


  %
  % validate
  %

  disp('Maximum error per parameter:');
  for p = 1:numel(params)
    disp(max(abs(result_der{p}(:) - true_param_der{p}(:))));

    figure(p), imagesc(result_der{p})
    title(sprintf('FMAD der #%i', p))
    figure(numel(params) + p), imagesc(true_param_der{p})
    title(sprintf('RMAD (backprop) #%i', p))
  end

end

