function training(experiment, varargin)
%TRAINING Performs training for any solver, model and dataset.
%   Name-value pairs are supported, listed below. See readme for examples.

  % options (override by calling script with name-value pairs).
  % (-D-) := if left empty, the default value for the chosen model will be used.
  opts.dataset = 'cifar';  % mnist, cifar or imagenet
  [opts, varargin] = vl_argparse(opts, varargin, 'nonrecursive');

  switch opts.dataset
  case 'mnist'
    opts.dataDir = [vl_rootnn() '/data/mnist'];  % data location
    opts.model = models.LeNet();  % choose model (type 'help models' for a list)
    numClasses = 10;
  case 'cifar'
    opts.dataDir = [vl_rootnn() '/data/cifar'];
    opts.model = models.BasicCifarNet('batchNorm', true);
    numClasses = 10;
  case {'imagenet', 'imagenet-100'}
    opts.dataDir = [vl_rootnn() '/data/ilsvrc12'];
    if strcmp(opts.dataset, 'imagenet-100')  % smaller imagenet subset
      numClasses = 100;
    else
      numClasses = 1000;
    end
    opts.model = models.VGG8('batchNorm', true, 'numClasses', numClasses);
  otherwise
    error('Unknown dataset.');
  end

  opts.resultsDir = [vl_rootnn() '/data/curveball/' opts.dataset '/' experiment];  % results location
  opts.conserveMemory = true;  % whether to conserve memory
  opts.numEpochs = [];  % epochs (-D-)
  opts.batchSize = [];  % batch size (-D-)
  opts.learningRate = [];  % learning rate (-D-)
  opts.weightDecay = 0;  % weight decay (-D-)
  opts.solver = solvers.SGD();  % solver instance to use (type 'help solvers' for a list)
  opts.gpu = 1;  % GPU index, empty for CPU mode
  opts.numThreads = 12;  % number of threads for image reading
  opts.augmentation = [];  % data augmentation (see datasets.ImageFolder) (-D-)
  opts.savePlot = true;  % whether to save the plot as a PDF file
  opts.continue = false;  % continue from last checkpoint if available
  opts.cifarShift = 2 ;  % number of pixels to shift in CIFAR data augmentation

  opts = vl_argparse(opts, varargin, 'nonrecursive');  % let user override options

  setup_curveball();  % add functions to path
  mkdir(opts.resultsDir);


  %
  % set up network
  %

  % use chosen model's output as the predictions
  assert(isa(opts.model, 'Layer'), 'Model must be a CNN (e.g. models.AlexNet()).')
  predictions = opts.model;

  % change the model's input name
  images = predictions.find('Input', 1);
  images.name = 'images';
  images.gpu = true;

  % validate the prediction size (must predict 1000 classes)
  defaults = predictions.meta;  % get model's meta information (default learning rate, etc)
  outputSize = predictions.evalOutputSize('images', [defaults.imageSize 5]);
  assert(isequal(outputSize, [1 1 numClasses 5]), 'Model output does not have the correct shape.');

  % mark conv layer that comes before the predictions, if possible
  if isequal(predictions.func, @vl_nnconv)
    prelayer = predictions.inputs{1};
    prelayer.name = 'prelayer';
  end

  % replace empty options with model-specific default values
  for name_ = {'numEpochs', 'batchSize', 'learningRate', 'weightDecay', 'augmentation'}
    name = name_{1};  % dereference cell array
    if isempty(opts.(name)) && isfield(defaults, name)
      opts.(name) = defaults.(name);
    end
  end

  % create losses
  labels = Input();
  obj = vl_nnloss(predictions, labels, 'loss', 'softmaxlog');

  % assign layer names automatically, and compile network
  Layer.workspaceNames();

  % compile net
  baseline = ~isa(opts.solver, 'CurveBall');
  if baseline  % standard net with loss
    output = obj;
  else  % net outputs the prediction only, loss is computed by hand
    output = predictions;
    if isequal(output.func, @vl_nnrelu)
      output = 1 * output;  % hack to keep ReLUs from being short-circuited in some models
    end
  end
  net = Net(output, 'conserveMemory', opts.conserveMemory);
  

  %
  % set up solver and dataset
  %
  
  % set solver learning rate
  solver = opts.solver;
  solver.learningRate = opts.learningRate(1) ;
  solver.weightDecay = opts.weightDecay ;
  if ~baseline
    solver.net = net;
  end
  
  % initialize dataset
  augmenter = @deal;
  switch opts.dataset
  case 'mnist'
    dataset = datasets.MNIST(opts.dataDir, 'batchSize', opts.batchSize);
    
  case 'cifar'
    dataset = datasets.CIFAR10(opts.dataDir, 'batchSize', opts.batchSize);
    % data augmentation: horizontal flips, and shifts of a few pixels
    shift = opts.cifarShift;
    augmenter = @(images) small_image_augmentation(images, shift);
    
  case {'imagenet', 'imagenet-100'}
    dataset = datasets.ImageNet('dataDir', opts.dataDir, ...
      'imageSize', defaults.imageSize, 'useGpu', ~isempty(opts.gpu));
    dataset.augmentation = opts.augmentation;
    dataset.numThreads = opts.numThreads;
    
    % smaller imagenet subset
    if strcmp(opts.dataset, 'imagenet-100')
      subset = dlmread('imagenet-subset.txt');
      imagenet_subset(dataset, subset);
    end
  end
  dataset.batchSize = opts.batchSize;

  stats = Stats();
  stats.registerVars({'obj', 'err', 'time'}, false);

  %
  % training
  %

  % continue from last checkpoint if there is one
  startEpoch = 1;
  if opts.continue
    [filename, startEpoch] = models.checkpoint([opts.resultsDir '/epoch-*.mat']);
  end
  if startEpoch > 1
    load(filename, 'net', 'stats', 'solver');
  end

  % enable GPU mode
  net.useGpu(opts.gpu);
  
  non_grad_params = [net.params([net.params.trainMethod] ~= 1).var];
  
  for epoch = startEpoch : opts.numEpochs
    % get the learning rate for this epoch, if there is a schedule
    if epoch <= numel(opts.learningRate)
      solver.learningRate = opts.learningRate(epoch);
    end

    % clear batch-norm moments derivatives for FMAD
    net.setDer(non_grad_params, repmat({0}, size(non_grad_params)));

    % training phase
    train_clock = tic();
    numBatches = floor(numel(dataset.trainSet) / opts.batchSize);
    for batch = dataset.train()
      % draw samples
      [images, labels] = dataset.get(batch);
      images = augmenter(images);  % apply data augmentation

      tic();
      % take one solver step
      if baseline
        % simple backprop
        net.eval({'images', images, 'labels', labels});

        solver.step(net);

        obj_value = net.getValue('obj');
      else
        % use CurveBall method. first, evaluate network on inputs
        net.eval({'images', images}, 'forward');

        % do a step
        solver.labels = labels;
        solver.step(net);

        obj_value = solver.last_loss;
      end

      % compute classification error and update statistics
      pred_value = net.getValue('predictions');
      assert(all(isfinite(pred_value(:))), 'Optimization diverged.');
      
      err_value = vl_nnloss(pred_value, labels, 'loss', 'classerror');
      stats.update('obj', obj_value, 'err', err_value);

      iter = stats.counts(1);  % number of iterations
      if iter > 3, stats.update('time', toc() * 1000); end  % skip first 3 batches

      if isa(solver, 'CurveBall') && solver.autolambda
        stats.update('lambda', solver.lambda, 'ratio', solver.ratio);
      end

      % report statistics and iteration number
      fprintf('ep%d %d/%d eta %s ', epoch, iter, numBatches, ...
        eta(stats.average('time'), iter, numBatches, epoch, opts.numEpochs));

      stats.print();
    end
    % push average objective and error (after one epoch) into the plot
    stats.update('total_time', toc(train_clock));
    stats.push('train');

    % validation phase
    numBatches = floor(numel(dataset.valSet) / opts.batchSize);
    for batch = dataset.val()
      [images, labels] = dataset.get(batch);

      % evaluate network
      tic();
      net.eval({'images', images}, 'test');

      % compute objective and error, and update statistics
      pred_value = net.getValue('predictions');
      obj_value = vl_nnloss(pred_value, labels, 'loss', 'softmaxlog');
      err_value = vl_nnloss(pred_value, labels, 'loss', 'classerror');
      stats.update('obj', obj_value, 'err', err_value, 'time', toc() * 1000);

      % report statistics and iteration number
      fprintf('val ep%d %d/%d ', epoch, stats.counts(1), numBatches);
      stats.print();
    end
    stats.push('val');

    % plot statistics
    stats.plot('figure', 1, 'names', {'obj', 'err'});

    if ~isempty(opts.resultsDir)
      % save the plot
      if opts.savePlot
        print(1, [opts.resultsDir '/plot.pdf'], '-dpdf');
      end

      % save checkpoint every few epochs
      if mod(epoch, 5) == 0
        save(sprintf('%s/epoch-%d.mat', opts.resultsDir, epoch), ...
          'net', 'stats', 'solver');
      end
    end
  end

  % save results
  if ~isempty(opts.resultsDir)
    save([opts.resultsDir '/results.mat'], 'net', 'stats', 'solver');
  end
end

function str = eta(batchTime, batch, batchesPerEpoch, epoch, numEpochs)
  % generate string estimating the time of completion
  completed = batch + batchesPerEpoch * (epoch - 1);
  total = batchesPerEpoch * numEpochs;
  
  secs = (total - completed) * batchTime / 1000 ;
  str = duration(0, 0, secs) ;
end
