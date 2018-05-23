function root = newton_root()
%NEWTON_ROOT Get the root path of the newton directory

	root = fileparts(fileparts(mfilename('fullpath'))) ;
