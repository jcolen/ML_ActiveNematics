# ML\_ActiveNematics

This is the official repository for the paper **Machine learning active-nematic hydrodynamics** (https://arxiv.org/abs/2006.13203).

Please direct all code-related questions to <jcolen@uchicago.edu>

# Parameter Estimation

Models can be trained for parameter estimation by running `train_parameter_estimator.py` in the `parameter_estimation` directory. This requires at least one argument designating the location of the training dataset. The default arguments produce the model configurations for 2D active nematics as described in the paper. Training can be performed by executing

    python train_parameter_estimator.py <DIRECTORY>

We found models for 3D active nematics performed best with 5x5x5 convolutional filters. Such models can be trained by executing

    python train_parameter_estimator.py <DIRECTORY> --kernel_size 5

This script produces a model file which will be saved in a `models` directory, and a set of predictions on the validation set which will be saved in the `predictions` directory.

# Time Evolution

Models for forecasting time evolution can be trained by running `train_frame_predictor.py` in the `time_evolution` directory. This requires at least one argument designating the location of the training dataset. The default arguments produce the model configurations for 2D active nematics as described in the paper. Training can be performed by executing

    python train_frame_predictor.py <DIRECTORY>

This script produces a model file which will be saved in a `models` directory. 

### Sharpening

The `time_evolution` folder also contains the code for our physically motivated sharpening algorithm. This is implemented in C and can be compiled using the included Makefile. The resulting libraries are imported and used to implement the full forecasting loop, the function `loop_frame_prediction` in `time_evolution/sharpen.py`.
