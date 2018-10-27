classdef CurveBall < solvers.Solver
  % CurveBall - Newton solver variant
  %
  % This variant works by 2 gradient descent updates: one for the linear
  % system (inversion), and another for the main objective.
  %
  % Assumptions:
  % - Network's output is a single tensor of predictions (not loss).
  % - Forward pass was executed just before calling this solver's step().

  properties
    loss = 'logistic'  % loss to use: 'ls' (least-squares), 'logistic' (softmax-log)

    lambda = 1  % Hessian regularization (0 = pure Newton)

    momentum = 0.9  % forget factor for linear system solution
    beta = 0.1  % learning rate for linear system GD step

    autoparam = true  % adapt momentum and beta
    autoparam_interval = 1  % only run autoparam every N steps. NOTE: autolambda depends on autoparam running in the same step.
    autoparam_print = false  % to print momentum/beta for debugging

    autolambda = true  % adapt lambda
    autolambda_w1 = 0.999  % factor, how much to change lambda each time
    autolambda_interval = 5  % only run autolambda every N steps. NOTE: autolambda depends on autoparam running in the same step.
    autolambda_thresh = 0.5  % rho thresholds to trigger lambda change; [low, high] vector, or scalar T to use [T, 2 - T]

    labels  % labels for this mini-batch (batch size assumed to be last dim.; cannot run with batch of 1)
    net  % pointer to network object

    count = 0  % iteration count

    state  % store for momentum (z)

    ratio = 0  % autolambda ratio is stored here for debugging
    last_loss  % last computed loss value
  end

  methods
    function o = CurveBall(varargin)
      % parse generic Solver arguments
      % default weight decay is 0, but can override with name-value pair
      varargin = o.parseGenericArgs([{'weightDecay', 0}, varargin]);

      % parse arguments specific to this solver
      vl_parseprop(o, varargin, {'loss', 'lambda', 'beta', 'momentum', ...
        'autoparam', 'autoparam_interval', 'autoparam_print', 'autolambda', ...
        'autolambda_w1', 'autolambda_thresh', 'autolambda_interval'});
      % don't let parent class clear some parameters
      o.conserveMemory = false;
      o.state = {};
    end

    function w = gradientStep(o, w, ~, lr, decay)
      % store properties in local vars
      net = o.net;  %#ok<*PROPLC>
      lambda = o.lambda;

      % initialize state if needed
      z = o.state;
      if isempty(z)
        z = cell(size(w));
        for i = 1:numel(w)
          z{i} = zeros(size(w{i}), 'like', w{i});
        end
      end

      use_decay = any(decay);
      assert(~use_decay, 'Not implemented.');
      

      %
      % do two GD steps. first update z (linear system state), then use the
      % result (whitened gradient estimate) to update the parameters w.
      %
      % zdelta = J' * Hl * J * z + lambda * z + J' * Jl'
      %
      % znew = momentum * z - beta * zdelta
      %
      % wnew = w + lr * znew
      %

      % compute factor Jz = J * z (for Hessian term)
      Jz = fmad(net, z);

      % evaluate loss, loss gradient Jl, and loss Hessian (Hl).
      % compute hessian_term = J' * Hl * Jz + Jl
      [Jz_, Jl, o.last_loss] = hessian_grad_loss(net, o.loss, o.labels, Jz);

      % backpropagate Jz_ + Jl
      hessian_term = backward(net, Jz_ + Jl);

      % compute z update (gradient descent on linear system)
      delta_z = cell(size(w));
      for i = 1:numel(w)
        delta_z{i} = hessian_term{i} + lambda * z{i};
      end

      
      %
      % automatic hyperparameters rho (momentum) and beta
      %

      if o.autoparam && rem(o.count, o.autoparam_interval) == 0
        % compute factor Jz = J * z (for Hessian term)
        Jdz = fmad(net, delta_z);

        % evaluate loss, loss gradient Jl, and loss Hessian (Hl).
        % compute hessian_term = J' * Hl * Jz
        Jdz_ = hessian_grad_loss(net, o.loss, o.labels, Jdz);

        % compute momentum and update by solving a 2x2 system A * x = b
        A11 = gather(Jdz(:)' * Jdz_(:));
        A12 = gather(Jz(:)' * Jdz_(:));
        A22 = gather(Jz(:)' * Jz_(:));

        b1 = gather(Jl(:)' * Jdz(:));
        b2 = gather(Jl(:)' * Jz(:));

        for i = 1:numel(w)
          % compute the system we want to invert
          z_vec = z{i}(:);
          dz_vec = delta_z{i}(:);

          % expand scalar if needed
          if isscalar(z_vec), z_vec(1:numel(dz_vec),1) = z_vec; end

          A11 = A11 + gather(dz_vec' * dz_vec) * lambda;  % accumulating scalars is faster (JIT)
          A12 = A12 + gather(dz_vec' * z_vec) * lambda;
          A22 = A22 + gather(z_vec' * z_vec) * lambda;
        end

        % compute beta and momentum coefficient
        A = double([A11, A12; A12, A22]);
        b = double([b1; b2]);
        m_b = A \ b;
        beta = m_b(1);
        momentum = -m_b(2);

        % sanity check
        if ~isfinite(momentum), momentum = 0; end
        if ~isfinite(beta), beta = 0; end

        % store values for next iterations if needed
        o.momentum = momentum;
        o.beta = beta;
      else
        momentum = o.momentum;
        beta = o.beta;
      end

      if o.autoparam_print
        fprintf('mom: %.1g beta: %.1g ', momentum, beta);
      end


      % update parameters with computed beta and momentum
      for i = 1:numel(w)
        % update linear system state
        z{i} = vl_taccum(momentum, z{i}, -beta, delta_z{i}) ;

        % update parameters (returned as this method's output)
        w{i} = vl_taccum(1, w{i}, lr(i), z{i}) ;
      end
      o.state = z;


      %
      % automatic lambda (trust region) hyperparameter
      %
      
      if o.autolambda && rem(o.count, o.autolambda_interval) == 0
        h_curr = o.last_loss ;
        M = -0.5 * m_b(:)' * b(:);

        % compute new value
        labels = o.labels;
        if isvector(labels)  % categorical labels
          batch_size = numel(labels);
        else  % regression targets, get last dimension
          batch_size = size(labels, ndims(labels));
        end

        % assign new parameter values now (don't wait for Solver.step)
        idx = [net.params.var];
        is_grad = ([net.params.trainMethod] == 1);
        net.setValue(idx(is_grad), w) ;

        % run network forward with new parameters
        net.eval({}, 'forward');

        % get value of last var (assumes only one output)
        pred = net.vars{end - 1};
        pred = reshape(pred, 1, 1, [], batch_size);  % reshape to 4D tensor for vl_nnloss

        % compute loss value
        switch o.loss
        case 'logistic'
          loss_value = vl_nnloss(pred, labels, 'loss', 'softmaxlog');
        otherwise
          error('Loss not yet supported with autolambda');
        end

        h_new = loss_value;

        % ratio between true curvature and quadratic fit curvature
        ratio = (h_new - h_curr) / M;

        % increase or decrease lambda based on ratio
        w1  = o.autolambda_w1 ^ o.autolambda_interval;
        thresh = o.autolambda_thresh;
        if isscalar(thresh)
          thresh = [thresh, 2 - thresh];
        end
        assert(diff(thresh) >= 0);

        if ratio < thresh(1)
          lambda = lambda / w1;
        elseif ratio > thresh(2)
          lambda = lambda * w1;
        end
        o.lambda = lambda;
        o.ratio = ratio;
      end

      o.count = o.count + 1;

    end

    function s = saveobj(o)
      % serialize to struct (called by the built-in function SAVE)
      % transfer state to CPU first
      s = o.saveGeneric();  % call parent class
      s.last_loss = gather(o.last_loss);
    end
  end

  methods (Static)
    function o = loadobj(s)
      % deserialize from struct (called by the built-in function LOAD)
      o = solvers.CurveBall();
      o.last_loss = s.last_loss;
      o.loadGeneric(s);  % call parent class

      % don't let parent class clear some parameters
      o.conserveMemory = false;
    end
  end
end


% computes the loss value, its gradient, and the hessian (multiplied by a
% vector x).
function [Hlx, Jl, loss_value] = hessian_grad_loss(net, loss, labels, x)
  if isvector(labels)  % categorical labels
    batch_size = numel(labels);
  else  % regression targets, get last dimension
    batch_size = size(labels, 4);
  end

  % get value of last var (assumes only one output)
  pred = net.vars{end - 1};
  pred = reshape(pred, 1, 1, [], batch_size);  % reshape to 4D tensor for vl_nnloss

  switch loss
  case 'ls'  % least-squares loss.
    % compute Hl * x. Hl = 2 / batch_size * I.
    Hlx = 2 / batch_size * x;

    % loss Jacobian, Jl
    assert(isequal(size(pred), size(labels)));
    Jl = 2 / batch_size * (pred - labels);

    % compute loss value
    loss_value = mean(sum((pred - labels).^2, 3), 4);

  case 'logistic'  % logistic loss.
    % compute Hl * x. for a single sample, Hl = diag(p) - p * p', where p
    % is a column-vector. for many samples, p has one column per sample,
    % and Hl is a block-diag with each block as above (so one independent
    % matrix-vec product per sample). first compute p' * x for all samples

    p = vl_nnsoftmax(pred);  % softmaxed probabilities

    if ismatrix(p)
      px = sum(p .* x, 1);
    else  % 4D tensor
      px = sum(p .* x, 3);
    end

    % now finish computing Hl * x = diag(p) * x - p * p' * x for every sample.
    Hlx = p .* x - p .* px;
    Hlx = Hlx / batch_size;

    if nargout <= 1, return, end  % early exit for single output


    % compute loss Jacobian, Jl. use backward pass of vl_nnloss
    Jl = vl_nnloss(pred, labels, single(1), 'loss', 'softmaxlog');  % loss gradient

    % compute loss value
    loss_value = vl_nnloss(pred, labels, 'loss', 'softmaxlog');

  otherwise
    error('Unknown loss.');
  end
end

