# DIMMPF
The code used to generate the results from our paper 'Differentiable Interacting Multiple Model Particle Filtering'.

## Experiment Arguments
The experiments in the paper can be run with main.py with the following arguments.
- `--help` List arguments.
- `--device` The device to store tensors on. [string] Options: **cuda|cpu**
- `--alg` The algorithm to use in the experiment, see the paper for details on each algorithm. [string] Options: **DIMMPF|DIMMPF-OT|DIMMPF-N|RLPF|Transformer|LSTM|IMMPF**
- `--lr` The learning rate parameter for the torch ADAM-W optimiser. [float]
- `--w_decay` The weight decay parameter for the torch ADAM-W optimiser. [float]
- `--lr_steps` The epoch numbers on which to decrease the ADAM-W learning rate. [list of ints]
- `--lr_gamma` The factor to multiply the ADAM-W learning rate at each step in `--lr_steps`. [float]
- `--clip` The value to clip the absolute value of the gradients to before updating parameters. Some algorithms clip the gradients passed between time-steps, this parameter does not affect that process. [float]
- `--lamb` The value of the coefficient λ, that is the ratio of the weight of the contributions of likelihood loss to the MSE loss to the combined training loss. [float]
- `--layers` For the DPF experiments, the number of hidden layers to use in the FCNNs that parameterise each model, ignored for non-DPF experiments. [int]
- `--hid_size` For the DPF experiments, the number of nodes in each hidden layer of the FCNNs that parameterise each model, ignored for non-DPF experiments. [int]
- `--epochs` The number of epochs to train for. [int]
- `--n_runs` The number of repeats to run. [int] 
- `--store_loc` The results will be saved as a python dictionary at `./results/[--store_loc].pickle`. [string]
- `--data_dir` The data created will be stored in `./data/[--data_dir]/`. [string]

## Project Structure
- `./main.py` Code to parse the user specified arguments and initialise an experiment.
- `./dpf-rs/` A python package for generic differentiable filtering.
- `./net.py` Implementations of the models used.
- `./SimulationRS.py` Contains the DIMMPF filtering algorithm.
- `./trainingRS.py` Contains the training loops and testing functions.
