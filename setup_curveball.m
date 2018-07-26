function setup_curveball()
%SETUP_NEWTON Sets up CurveBall, by adding it to the path

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/fmad'], [root '/fmad/mex'], ...
    [root '/models'], [root '/utils'], [root '/tests']) ;

  % check for AutoNN
  if ~exist('Layer', 'class')
    vl_contrib('setup', 'autonn');
  end

end
