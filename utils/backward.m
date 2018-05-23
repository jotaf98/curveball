function dW = backward(net, dY)
% backward pass, prediction derivatives to parameter derivatives.
% NOTE: assumes forward() was called on same data; or won't behave as
% expected.

  net.eval({}, 'backward', dY);
  idx = [net.params([net.params.trainMethod] == 1).var] ;
  dW = net.getDer(idx);
	if isscalar(idx), dW = {dW} ; end
end
