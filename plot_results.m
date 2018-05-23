function plot_results(varargin)
%PLOT_RESULTS Plots all results for a given dataset.
%   Name-value pairs are supported, listed below.
%   E.g.: plot_results('dataset', 'imagenet', 'set', 'val')

  opts.dataset = 'cifar';  % choose dataset
  opts.set = 'train';  % train or val
  opts.var = 'err';  % err or obj (error/objective)
  opts.save = false;  % if true, saves as a PDF file in the base directory
  
  opts.baseDir = [vl_rootnn() '/data/curveball/<dataset>'];  % results location
  opts.subdir = '';  % optional subdirectory
  
  opts = vl_argparse(opts, varargin);
  
  % compose directory where experiments are located
  base = strrep(opts.baseDir, '<dataset>', opts.dataset);
  base = [base '/' opts.subdir];
  
  % list experiments (folders)
  folders = dir(base);
  
  figure(10);
  clf();
  hold('on');
  ax = gca();
  set(ax, 'lineStyleOrder', {'-','--','-.',':','*','+'});
  
  for i = 1:numel(folders)
    name = folders(i).name;
    
    if folders(i).isdir && ~strncmp(name, '.', 1)
      % try to load final results first
      filename = [base '/' name '/results.mat'];
      if ~exist(filename, 'file')
        % if it doesn't exist, look for the most recent checkpoint
        filename = models.checkpoint([base '/' name '/epoch-*.mat']);
      end
      
      if ~isempty(filename)
        % load result statistics
        data = load(filename, 'stats');

        % get the relevant curve
        values = data.stats.values(opts.set, opts.var);
        
        % display it
        plot(values, 'DisplayName', name, 'LineWidth',2);
      end
    end
  end
  
  % make plots look nice
  legend('show');
  grid('on');
  
  xlabel('Epochs');
  ylabel([opts.set ' ' opts.var]);
  
  % save to PDF
  if opts.save
    filename = [base '/results.pdf'];
    print(10, filename, '-dpdf');
  end
end
