

# Machine 1:
## Folder construct:
  ```collect_data.m``` - collect data of interest from the simulation data

  ```mac1.py``` - implementation for initialization, data loading, data pre-processing, training and testing

  ```mac1_model.py``` - model definition

  ```mac1_tuning.ipynb``` - jupyter lab script for training scheme and evaluation

## Data Format:
Example: data_mac1_full_wBus2.mat

The mat file should contains a struct named data, and data includes all the data that the model may need: [bus\_v, cur, mac\_and, mac_spd, pelect, pmech, qelect] at machine 1. Also, [bus\_v] at bus 2 is also included and saved as in column [bus2]. Some extra information is included such as length of the sequences and the filename of where the data come from.

There are 5277 available data files (excluding those with faults on line 5). I use the first 4500 for train, and 4800-5200 for testing.

## Details
wandb(https://wandb.ai/) is a library that helps me to keep track of model performance and hyperparameter settings. You may ignore any wandb-related statements if you are not using it.

This model uses a simplified data set that is built from the original data set, to reduce data loading time. For larger models, you may want to load data from the original data set directly so you are more flexible to load specific data.

## Model definition
```
mac1 = Machine1(n_layers=2, hidden_dim=[256,256], n_training_sample=1600, data_len=700, batchsize=100, data_interpolation_rate=5)
```
```n_layers``` is the number of LSTM layers and ```hidden_dim``` specifies the number of LSTM cells in each layer.

```n_training_sample``` sets how many training samples are used. To speed up, only 1600/4500 samples are used in develop. 

```data_len``` is the length of training sequences. To fully utilize the data, it should be ~1000. Unfortunately, in my desktop, it is limited to 700 because it is constrained by my GPU memory. ```data_interpolation_rate``` means interpolation by 5x. In this case, time interval between 2 consecutive data samples over time decreases from 100ms to 20ms.
When it comes to evaluation/production, you may want to use as many data as possible.



