# Time Evolution

Models for forecasting time evolution can be trained by running `train_frame_predictor.py`. This requires at least one argument designating the location of the training dataset. The default arguments produce the model configurations for 2D active nematics as described in the paper. Training can be performed by executing

    python train_frame_predictor.py <DIRECTORY>

This script produces a model file which will be saved in a `models` directory. 

### Sharpening

The subfolder `sharpening` contains the code for our physically motivated sharpening algorithm. The bulk of this is implemented in C and can be compiled using the included Makefile. The resulting libraries can be imported and used as in `sharpen.py`.

### Simulation datasets

The file `simulation_datasets.py` is designed to handle the director field images produced by the hybrid Lattice Boltzmann simulations used in this study. These datasets start with a root directory and obtain labeled data from all valid subdirectories (i.e. containing correct simulation data) within that root directory. Labels are obtained by tokenizing the subdirectory names (with underscore \_ delimiters). For example, a directory like "Z\_0.02" will be labeled with the dictionary `{'z': 0.02}`. A directory like "K\_0.1\_Z\_0.02" will be labeled `{'k': 0.1, 'z':0.02}`.

