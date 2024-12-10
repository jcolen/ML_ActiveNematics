# ML_ActiveNematics

This is the official repository for the paper [Machine learning active-nematic hydrodynamics](https://arxiv.org/abs/2006.13203).

## Parameter estimation

Models can be trained for parameter estimation by running `parameter_estimation/src/train_parameter_estimator.py`. 
The script takes several command line arguments. The most important one is `--directory` which specifies the dataset location. 
The script produces model weights and predictions which are stored in a common directory specified by the `--save_name` argument. 
Sample trained models are in `parameter_estimation/models` and notebooks containing prediction plots are in `parameter_estimation/notebooks`. 

## Time evolution

Models for forecasting time evolution can be trained by running `time_evolution/src/train_frame_predictor.py`.
The script produces model weights which are stored in a directory specified by the `--save_name` argument. 
Sample trained models are in `time_evolution/models` and notebooks containing prediction plots are in `time_evolution/notebooks`. 

### Sharpening

The `time_evolution/src` folder also contains code for our physically motivated sharpening algorithm. This is implemented in C and can be compiled using the included Makefile. The resulting libraries are imported and used to implement the full forecasting loop, the function `loop_frame_prediction` in `time_evolution/src/sharpen.py`.  

## Correction note

We recently submitted a correction after observing significant stochastic and unpredictable behavior during both training and inference for both model types - the parameter estimator and the time evolution forecaster.
This repository has been updated to address and fix those issues so that the scripts produce consistent and accurate model behavior. 
The wikipage contains a more detailed overview of our investigation into what the issues are, why they occurred, and how to fix them.