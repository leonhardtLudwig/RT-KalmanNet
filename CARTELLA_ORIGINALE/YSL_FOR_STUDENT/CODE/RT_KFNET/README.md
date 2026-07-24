## RT - KalmanNet - SHORT DESCRIPTION

## The important (newly implemented/modified) files for the project are:

# main_Robust_KNet.ipynb
    - It is the main jupyter notebook used for plotting and evaluating the different types of filters implemented

# RobustKalmanPY/robust_kalman.py
    - Contains the REKF translated code from MATLAB and the RT-KalmanNet 
    
# KNet/RT_KalmanNet_nn.py
    - Contains the implemented Neural Network structure used in the robust_kalmanet.py filter

# Simulations/Synthetic_NL_model/parameters.py
    - Contains the definition of the Synthetic Non-Linear model used for evaluating the filter

# Simulations/Lorenz_Atractor/parameters.py
    - Contains the definition of the Lorenz-Atractor model used for evaluating the filter
    - This file derives from the original paper file but is modified to contain the no_batch implementations of functions f(x) and h(x)
## 
## 
## 

## KalmanNet 

## Feb.13, 2023 Update "batched"

Support a batch of sequences being processed simultaneously, leading to dramatic efficiency improvement.

## Link to paper

[KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics](https://arxiv.org/abs/2107.10043)

## Running code

This branch simulates architecture #2 in our paper. There are main files simulating the linear and non-linear cases respectively.

* Linear case (canonical model or constant acceleration model)

```
python3 main_linear_canonical.py
python3 main_linear_CA.py
```

* Non-linear Lorenz Attractor case (Discrete-Time, decimation, or Non-linear observation function)

```
python3 main_lor_DT.py
python3 main_lor_decimation.py
python3 main_lor_DT_NLobs.py
```

## Parameter settings

* Simulations/model_name/parameters.py

Contain model settings: m, n, f/F, h/H, Q and R. 

* Simulations/config.py

Contain dataset size, training parameters and network settings.

* main files

Set flags, paths, etc.


