function Y = forward(net, W, outputPosition)
% forward pass, parameters to predictions.
% if there are multiple outputs, outputPosition selects which one.

  if nargin < 3
    outputPosition = 1;
  end

  if nargin > 1 && ~isempty(W)
    params = [net.params([net.params.trainMethod] == 1).var];
    W_ = net.getValue(params);
    net.setValue(params, W);  % override W
  end

  net.eval({}, 'forward');

  % grab output value
  if isequal(net.forward(end).func, @root)
    % "root" captures multiple output values, select one
    Y = net.getValue(net.forward(end).inputVars(outputPosition));
  else
    Y = net.getValue(net.forward(end).outputVar);  % single value
  end
  
  if nargin > 1 && ~isempty(W)
    net.setValue(params, W_);  % restore W
  end
end
