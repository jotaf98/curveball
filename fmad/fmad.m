function der = fmad(net, dW)
%   FMAD Computes the forward-order auto-diff (FMAD) of a network for the
%   given parameter derivatives, and input sizes.


  % get variable values (including parameters and inputs), and clear all
  % derivatives (even-numbered vars)
  vars = net.vars;
  vars(2:2:end) = {[]};

  % assign projected derivatives of parameters
  params = [net.params.var];  % list all parameters
  is_grad = ([net.params.trainMethod] == 1);
  vars(params(is_grad) + 1) = dW;  % assign dW only to gradient-based params
  vars(params(~is_grad) + 1) = {0};  % assign 0 to others (e.g. bnorm moments)

  % we do not differentiate w.r.t. inputs, so their derivatives are 0.
  % we need to initialize with the same size as the input's value (i.e.,
  % scalar 0 won't do), for matrix multiplications to yield correct sizes.
  names = fieldnames(net.inputs);
  for i = 1:numel(names)
    var_idx = net.inputs.(names{i});
    vars{var_idx + 1} = zeros(size(vars{var_idx}), 'like', vars{var_idx});
  end


  % iterate layers in forward order
  forward = net.forward;

  for k = 1:numel(forward)
    % fill in arguments list for this layer (args), by grabbing any
    % computed variables' values from the vars cell array
    layer = forward(k);
    args = layer.args;
    args(layer.inputArgPos) = vars(layer.inputVars);

    % create a similar list, but with the corresponding derivatives
    args_der = cell(size(args));
    args_der(:) = {0};
    args_der(layer.inputArgPos) = vars(layer.inputVars + 1);


    assert(~any(cellfun('isempty', args_der(layer.inputArgPos))));  % sanity check

    % do this layer's forward-mode auto-diff operation on those arguments
    assert(~isempty(layer.func), 'Unknown layer type.');

    switch func2str(layer.func)

    case 'mtimes'
      % matrix multiplication: Y = A * B  ->  dY = A * dB + dA * B
      A = args{1};
      B = args{2};
      dA = args_der{1};
      dB = args_der{2};
      der = A * dB + dA * B;

    case 'vl_nnwsum'
      % weighted sum: Y = w(1) * A + w(2) * B + ...
      %  ->  dY = w(1) * dA + w(2) * dB + ...
      w = args{end};
      dA = args_der(1:end-2);

      der = w(1) * dA{1};
      for j = 2:numel(dA)
        der = der + w(j) * dA{j};
      end

    case 'slice_wrapper'
      % slice the derivative using the same indices as the forward pass
      % i.e  if Y = X(SUBS), then DER = dX(SUBS)
      der = slice_wrapper(args_der{1}, args{2});

    case 'cat'
      % concatente the incoming differentials to match the forward OP,
      % replacing empty differentials with zeros of the same size
      differentials = args_der(2:end) ;
      for ii = 1:numel(differentials)
        if isequal(differentials{ii}, 0)
          A = args{ii+1} ; % finding corresponding input argument
          differentials{ii} = zeros(size(A), 'like', A) ;
        end
      end
      der = cat(args{1}, differentials{:}) ;

    case 'bsxfun'
      % support axis expansion for @times and @power operators. The output
      % takes the generic form Y = A @OP B, but the form of the derivative
      % depends on the specific @OP.
      % If @OP = @TIMES, then Y = A * B -> dY = A * dB + dA * B (as above)
      % If @OP = @POWER, then
      %   Y = A ^ B -> dY = [ln(A) * (A^B) * dB] + [dA * B * A^(B-1)]
      op = args{1} ;
      A = args{2} ;
      B = args{3} ;
      dA = args_der{2} ;
      dB = args_der{3} ;
      % for each @OP, we must take care to avoid a dimension collapse in DER
      % that can occur by summing over empty input differentials (i.e. when
      % either dA or dB are empty).
      switch func2str(op)
        case 'times'
          der = zeros(1, 'like', A) ;
          der = der + dA .* B ;
          der = der + dB .* A ;
        case 'power'
          der = zeros(1, 'like', A) ;
          der = der + dA .* B .* (A .^ (B - 1)) ;
          if ~all(dB(:) == 0)
            error('only constant powers are currently supported for bsxfun') ;
            % note: only supports constant powers, otherwise log will break
            % on the GPU
            der = der + log(A) .* (A .^ B) .* dB ; % dream outcome
          end
        otherwise, error('%s not yet supported\n', func2str(op)) ;
      end

    case {'vl_nnrelu', 'vl_nnsigmoid', 'vl_nntanh', 'vl_nnmask'}
      % for element-wise functions, we just need to call their standard
      % back-prop operation.
      % e.g.: Y = vl_nnrelu(X)  ->  dX = vl_nnrelu(X, dY)
      der = layer.func(args{1}, args_der{1}, args{2:end});

    case 'vl_nndropout_wrapper'
      % similar to the above, but with extra fixed arguments, which are
      % not differentiable. Y = vl_nndropout_wrapper(X, M, T)
      %  ->  dX = vl_nndropout_wrapper(X, M, T, dY)
      der = vl_nndropout_wrapper(args{1:3}, args_der{1});

    case 'vl_nnconv'
      % convolution, generalizes matrix multiplication above.
      % add the bias derivative, but only on one of the convs (otherwise it
      % would be doubled).
      x = args{1};
      w = args{2};
      dx = args_der{1};
      dw = args_der{2};
      db = args_der{3};
      if isequal(db, 0)
        db = [];
      end
%       der = vl_nnconv(x, dw, [], args{4:end}) + ...
%             vl_nnconv(dx, w, db, args{4:end});

      der = vl_nnconv(cat(3, x, dx), cat(3, dw, w), db, args{4:end});

    case 'vl_nnpool'
      % average and max pooling
      o.stride = 1;
      o.pad = 0;
      o.method = 'max';
      o = vl_argparse(o, args(3:end));

      switch o.method
      case 'avg'
        % just average-pool the input derivatives (linear operation)
        der = vl_nnpool(args_der{1}, args{2:end});

      case 'max'
        % get pooling switches from forward operation, and use them to grab
        % the same elements from the input derivatives
        pool_size = args{2};
        idx = pool_switches(args{1}, pool_size, o.stride, o.pad);
        der = args_der{1}(idx);

      otherwise
        error('Unknown pooling method.');
      end

    case 'vl_nnmaxout'
        sz        = size(args{1});
        blocks    = sz(3) / args{2} ;
        idx       = pool_switches(reshape(args{1}, prod(sz(1:2)), sz(3), []), [1, args{2}], [1,args{2}], 0);
        der       = reshape(args_der{1}(idx), [sz(1:2), blocks, sz(4:end)]);

    case 'vl_nnbnorm_wrapper'
      % batch normalization
      x = args{1} ; g = args{2} ; %b = args{3} ;
      dx = args_der{1} ; dg = args_der{2} ; db = args_der{3} ;

      % extend scalars
      if isscalar(g), g(1:size(x,3)) = g; end
      %if isscalar(b), b(1:size(x,3)) = b; end
      if isscalar(dg), dg(1:size(x,3)) = dg; end
      if isscalar(db), db(1:size(x,3)) = db; end
      if isscalar(dx), dx = dx + zeros(size(x), 'like',x); end

      %der = fmad_bnorm(x, g, b, dx, dg, db) ;
      der = vl_nnfmadbnorm(x, g, dx, dg, db) ;

      % store debug values
      %tmp = struct() ;
      %vars = {'x', 'g', 'b', 'dx', 'dg', 'db', 'der2'} ;
      %for ii = 1:numel(vars)
      %  tmp.(vars{ii}) = gather(eval(vars{ii})) ;
      %end
      %save('bnorm_vars.mat', '-struct', 'tmp') ;

      % kept previous version for comparison
      %der = bnorm_forward_der_prev(args, args_der) ;


    otherwise
      error('Unknown layer type %s.', func2str(layer.func));
    end

    % store output into the derivative of the output var
    assert(isscalar(layer.outputVar), 'Multiple outputs not supported.');
    vars{layer.outputVar + 1} = der;
  end
end
