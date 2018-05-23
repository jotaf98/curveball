
refresh = false ;
if ~exist('der2', 'var') || refresh
	tmp = load('bnorm_vars.mat') ;
	x = gpuArray(tmp.x) ;
  g = gpuArray(tmp.g) ;
  b = gpuArray(tmp.b) ;
	dx = gpuArray(tmp.dx) ;
  dg = gpuArray(tmp.dg) ;
  db = gpuArray(tmp.db) ;
  der2 = gpuArray(tmp.der2) ;
end

der = fmad_bnorm(x, g, b, dx, dg, db) ;
sum(der(:))
diff = sum(abs(der(:) - der2(:))) ;
rel_error = diff / norm(der(:)) ;
fprintf('rel error %g\n', rel_error) ;
