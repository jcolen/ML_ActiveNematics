# Parameter Estimation

Models can be trained for parameter estimation by running `train_parameter_estimator.py`. This requires at least one argument designating the location of the training dataset. The default arguments produce the model configurations for 2D active nematics as described in the paper. Training can be performed by executing

    python train_parameter_estimator.py <DIRECTORY>

We found models for 3D active nematics performed best with 5x5x5 convolutional filters. Such models can be trained by executing

    python train_parameter_estimator.py <DIRECTORY> --kernel_size 5

This script produces a model file which will be saved in a `models` directory, and a set of predictions on the validation set which will be saved in the `predictions` directory.

### Simulation datasets

The file `simulation_datasets.py` is designed to handle the director field images produced by the hybrid Lattice Boltzmann simulations used in this study. These datasets start with a root directory and obtain labeled data from all valid subdirectories (i.e. containing correct simulation data) within that root directory. Labels are obtained by tokenizing the subdirectory names (with underscore \_ delimiters). For example, a directory like "Z\_0.02" will be labeled with the dictionary `{'z': 0.02}`. A directory like "K\_0.1\_Z\_0.02" will be labeled `{'k': 0.1, 'z':0.02}`.
