## CurveBall

<img alt="Sandy Koufax" src="http://farm4.static.flickr.com/3271/3050357231_e923027b97_o.gif">

This is the accompanying code repository for the paper:

João F. Henriques, Sebastien Ehrhardt, Samuel Albanie, Andrea Vedaldi  
**["Small steps and giant leaps: Minimal Newton solvers for Deep Learning"](https://arxiv.org/abs/1805.08095)**  
arXiv preprint, 2018


### Warning

This code is undergoing refactoring, which may introduce subtle bugs. Also, be aware that our implementation of forward-mode automatic differentiation (FMAD) could be more efficient, when compared to standard forward/back-propagation operations (CuDNN). We expect to improve this over time.


### Installation

Requirements:
- A recent Matlab installation (at most 2 years old).
- The latest master version of [MatConvNet](https://github.com/vlfeat/matconvnet) on GitHub.
- [AutoNN](https://github.com/vlfeat/autonn) (can be installed in the Matlab console with `vl_contrib install autonn`).

For speed, the forward-mode automatic differentiation (FMAD) is not all pure Matlab, but uses a couple of custom CUDA kernels (batch-norm and max-pooling switches). This requires compilation, by calling `compile_fmad`.


### Training

The main function is called `training`. It supports the models (VGG/AlexNet/ResNet/etc), datasets (MNIST/CIFAR/ImageNet) and solvers (SGD/Adam/etc) defined in AutoNN. It also supports our new solver, called `CurveBall`. Note that not all models may work (due to undefined ops in the FMAD routine).

The first argument is an experiment name (subdirectory to store results), followed by name-value pairs. By default, the results are stored in `<matconvnet>/data/curveball`. The datasets are downloaded by default to `<matconvnet>/data/<datasetname>`. These can be overriden, but it's perhaps more practical to symlink `data` to a desired data folder. One important parameter is the `'gpu'`, which defines the GPU index to use for training. By default the first GPU is used.

The full parameter list is at the top of the `training.m` file. A few examples follow.

- Basic CIFAR CNN:  
`training('basic-curveball', 'solver',CurveBall(), 'learningRate',1)`

- Basic CIFAR CNN with Adam baseline:  
`training('basic-adam', 'solver',solvers.Adam(), 'learningRate',0.001)`

- Basic CIFAR CNN without batch-norm:  
`training('basic-nobatchnorm-curveball', 'solver',CurveBall('lambda',10), 'learningRate',1, 'model',models.BasicCifarNet('batchNorm',false))`

- ResNet-18 with dropout:  
`training('resnet18-dropout0.3-curveball', 'solver',CurveBall(), 'learningRate',1, 'model',cifar_resnet('dropout',0.3))`

- VGG-f on ImageNet-100:  
`training('vggf-curveball', 'dataset','imagenet-100', 'solver',CurveBall(), 'learningRate',1)`

Results for a given dataset can be plotted together and compared using `plot_results`.
