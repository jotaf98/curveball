function compile_fmad(varargin)
% COMPILE_NEWTON compile the C++/CUDA components of the newton module

tokens = {vl_rootnn, 'matlab', 'mex', '.build', 'last_compile_opts.mat'} ;
last_args_path = fullfile(tokens{:}) ; opts = {} ;
if exist(last_args_path, 'file'), opts = {load(last_args_path)} ; end
opts = selectCompileOpts(opts) ;
%vl_compilenn(opts{:}, varargin{:}, 'preCompileFn', @preCompileFn, 'cudaMethod', 'nvcc') ;
vl_compilenn(opts{:}, varargin{:}, 'preCompileFn', @preCompileFn) ;

% -------------------------------------
function opts = selectCompileOpts(opts)
% -------------------------------------
% really need a better fix at some point
keep = {'enableGpu', 'enableImreadJpeg', 'enableCudnn', 'enableDouble', ...
        'imageLibrary', 'imageLibraryCompileFlags', ...
        'imageLibraryLinkFlags', 'verbose', 'debug', 'cudaMethod', ...
        'cudaRoot', 'cudaArch', 'defCudaArch', 'cudnnRoot', 'preCompileFn'} ;
s = opts{1} ;
f = fieldnames(s) ;
for i = 1:numel(f)
  if ~ismember(f{i}, keep)
    s = rmfield(s, f{i}) ;
  end
end
opts = {s} ;

% -----------------------------------------------------------------------------
function [opts, mex_src, lib_src, flags] = preCompileFn(opts, mex_src, lib_src, flags)
% -----------------------------------------------------------------------------
  mcn_root = vl_rootnn() ;
  root = fullfile(fileparts(mfilename('fullpath')), 'fmad') ;

  % Build inside the module path
  flags.src_dir = fullfile(root, 'src') ;
  flags.mex_dir = fullfile(root, 'mex') ;
  flags.bld_dir = fullfile(flags.mex_dir, '.build') ;
  implDir = fullfile(flags.bld_dir,'bits/impl') ;
  if ~exist(implDir, 'dir'), mkdir(implDir) ; end
  mex_src = {} ; lib_src = {} ;
  if opts.enableGpu, ext = 'cu' ;  else, ext = 'cpp' ; end

  % Add mcn dependencies
  lib_src{end+1} = fullfile(mcn_root, 'matlab/src/bits', ['data.' ext]) ;
  lib_src{end+1} = fullfile(mcn_root, 'matlab/src/bits', ['datamex.' ext]) ;
  lib_src{end+1} = fullfile(mcn_root,'matlab/src/bits/impl/copy_cpu.cpp') ;

  if opts.enableGpu
    lib_src{end+1} = fullfile(mcn_root,'matlab/src/bits/datacu.cu') ;
    lib_src{end+1} = fullfile(mcn_root,'matlab/src/bits/impl/copy_gpu.cu') ;
  end

  % ----------------------
  % include required files
  % ----------------------
  % old style include - leave as comment in case users have different setup
  %if ~isfield(flags, 'cc'), flags.cc = {inc}; else, flags.cc{end+1} = inc; end

  % new style include
  inc = sprintf('-I"%s"', fullfile(mcn_root,'matlab','src')) ;
  inc_local = sprintf('-I"%s"', fullfile(root, 'src')) ;
  flags.base{end+1} = inc ; flags.base{end+1} = inc_local ;

  % Add module files
  lib_src{end+1} = fullfile(root,'src','bits',['nnfmadbnorm.' ext]) ;
  mex_src{end+1} = fullfile(root,'src',['vl_nnfmadbnorm.' ext]) ;
