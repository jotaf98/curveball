function y = vl_nntanh(x,dzdy)
%VL_NNTANH CNN hyperbolic tangent nonlinear unit.

if nargin <= 1 || isempty(dzdy)
  y = tanh(x) ;
else
  y = (1 - tanh(x).^2) .* dzdy ;
end
