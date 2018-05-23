function imagenet_subset(dataset, subset)
% IMAGENET_SUBSET Modifies a ImageNet dataset to use a subset of classes

  rng(0);
  if isnumeric(subset)
    if isscalar(subset)
      % pick random subset with given size
      idx = randperm(1000, subset);
    else  % passed it in directly
      idx = subset;
    end
  else
    % named subsets
    switch subset
    case 'fruits'
      idx = [323, 320, 324, 229, 322, 326, 321, 318, 331, 319];
    case 'animals'
      idx = [1347, 1123, 1161, 1276, 1215];
    otherwise
      error('Unknown subset.');
    end
  end
  
  % index to synset mapping
  synsets = [];
  load([dataset.dataDir '/ILSVRC2014_devkit/data/meta_clsloc.mat'], 'synsets');
  
  if ischar(idx)
    % find classes by name
    subset_names = strtrim(strsplit(lower(idx), ','));
    synset_names = {synsets.words};
    
    valid = false(numel(synset_names), 1);
    for i = 1:numel(synset_names)
      % each synset name contains multiple synonims, separated by commas
      synonyms = strtrim(strsplit(lower(synset_names{i}), ','));
      valid(i) = any(ismember(subset_names, synonyms));
    end
    
    idx = find(valid);
    assert(numel(idx) == numel(subset_names), 'At least one class was not found.');
  end
  
  % synset IDs (e.g. 'n01542...', a string) of the selected classes
  synset_ids = {synsets(idx).WNID};

  % binary vector of which synsets/classes to keep
  valid_classes = ismember(dataset.classes.name, synset_ids);
  
  assert(nnz(valid_classes) == numel(synset_ids), 'At least one class was not found.');
  
  % remove all other classes, and their samples
  invalid_samples = ~ismember(dataset.labels, find(valid_classes));
  dataset.labels(invalid_samples) = [];
  dataset.sets(invalid_samples) = [];
  dataset.filenames(invalid_samples) = [];
  dataset.augmentImage(invalid_samples) = [];
  
  dataset.trainSet = find(dataset.sets == 1);
  dataset.valSet = find(dataset.sets == 2);
  
  % remap labels to 1:N
  map = cumsum(double(valid_classes));
  dataset.labels = map(dataset.labels);
end

