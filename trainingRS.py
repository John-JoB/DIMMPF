import torch as pt
from dpf_rs.simulation import Differentiable_Particle_Filter
from tqdm import tqdm
from typing import Iterable
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy
from dpf_rs.loss import Loss, Compound_Loss, Magnitude_Loss
from dpf_rs import model
from dpf_rs import loss as losses
from dpf_rs import results
import time
from dpf_rs.utils import fix_rng


def _test(
        DPF: Differentiable_Particle_Filter, 
        loss: Loss, 
        T: int, 
        data: pt.utils.data.DataLoader,
        scale = None
        ):
    DPF.eval()
    try:
        IMMtype = DPF.IMMtype
        if IMMtype == 'new':
            DPF.IMMtype = 'normal'
    except:
        pass
    with pt.inference_mode():
        for i, simulated_object in enumerate(data):
            if not scale is None:
                simulated_object.state -= scale[0]
                simulated_object.observations -= scale[2] 
                simulated_object.state /= scale[1]
                simulated_object.observations /= scale[3]
            loss.clear_data()
            loss.register_data(truth=simulated_object)
            DPF(simulated_object, T, loss.get_reporters())
            loss_t = loss.per_step_loss() 
            if not scale is None:
                loss_t = loss_t * (scale[1][0]**2)
            loss_t = loss_t.to(device ='cpu').detach().numpy()
    print(f'Test loss: {np.mean(loss_t)}')
    try:
        DPF.IMMtype = IMMtype
    except:
        pass
    return np.array([np.mean(loss_t)]), np.mean(loss_t, axis = 0)



def test(DPF: Differentiable_Particle_Filter,
        loss: Loss, 
        T: int, 
        data: pt.utils.data.Dataset, 
        batch_size:int, 
        fraction:float, 
        ):
        if fraction == 1:
            test_set = data
        else:
            test_set, _ =  pt.utils.data.random_split(data, [fraction, 1-fraction])
        if batch_size == -1:
            batch_size = len(test_set)
        test = pt.utils.data.DataLoader(test_set, min(batch_size, len(test_set)), shuffle=False, collate_fn=data.collate, num_workers= data.workers, drop_last=True)
        start = time.time()
        results = _test(DPF, loss, T, test)
        return *results, np.array([0]), np.array([time.time()-start])


def e2e_train(
        DPF: Differentiable_Particle_Filter,
        DPF_redefined: Differentiable_Particle_Filter,
        opt: pt.optim.Optimizer,
        loss: Loss, 
        T: int, 
        data_train: pt.utils.data.Dataset,
        data_test: pt.utils.data.Dataset,
        batch_size: Iterable[int], 
        set_fractions: Iterable[float], 
        epochs: int,
        test_scaling: float=1,
        opt_schedule: pt.optim.lr_scheduler.LRScheduler=None,
        verbose:bool=True,
        clip:float = pt.inf,
        normalise = True,
        lam = 0.02
        ):
    if data_test is None:
        train_set, valid_set, test_set = pt.utils.data.random_split(data_train, set_fractions)
    else:
        train_set = data_train
        valid_set, test_set = pt.utils.data.random_split(data_test, set_fractions)
    print(len(valid_set))
    if batch_size[0] == -1:
        batch_size[0] = len(train_set)
    if batch_size[1] == -1:
        batch_size[1] = len(valid_set)
    if batch_size[2] == -1:
        batch_size[2] = len(test_set)

    train = pt.utils.data.DataLoader(train_set, batch_size[0], shuffle=True, collate_fn=data_train.collate, num_workers= data_train.workers)
    valid = pt.utils.data.DataLoader(valid_set, min(batch_size[1], len(valid_set)), shuffle=False, collate_fn=data_train.collate, num_workers= data_train.workers)
    test = pt.utils.data.DataLoader(test_set, min(batch_size[2], len(test_set)), shuffle=False, collate_fn=data_train.collate, num_workers= data_train.workers, drop_last=True)
    times = np.empty(epochs)

    train_loss = np.zeros(len(train)*epochs)
    test_loss = np.zeros(epochs)
    min_valid_loss = pt.inf
    if normalise:
        for i, simulated_object in enumerate(train):
            if i == 0:
                mean_state = pt.mean(simulated_object.state, dim=(0,1))
                mean_sq_state = pt.mean(simulated_object.state **2, dim=(0,1))
                mean_obs = pt.mean(simulated_object.observations, dim=(0,1))
                mean_sq_obs = pt.mean(simulated_object.observations **2, dim=(0,1))
            else:
                mean_state += pt.mean(simulated_object.state, dim=(0,1))
                mean_sq_state += pt.mean(simulated_object.state **2, dim=(0,1))
                mean_obs += pt.mean(simulated_object.observations, dim=(0,1))
                mean_sq_obs += pt.mean(simulated_object.observations **2, dim=(0,1))
        mean_state = mean_state/len(train)
        sd_state = pt.sqrt(mean_sq_state/len(train) - mean_state**2)
        mean_obs = mean_obs/len(train)
        sd_obs = pt.sqrt(mean_sq_obs/len(train) - mean_obs**2)

    if lam != 0:
        likelihood_loss = Magnitude_Loss(results.Log_Likelihood_Factors(), sign=-1)
        complete_loss = Compound_Loss([loss, likelihood_loss])
    else:
        complete_loss = loss
    
    for epoch in range(epochs):
        start_ep = time.time()
        DPF.train()
        train_it = enumerate(train)
        for b, simulated_object in train_it:
            if not normalise:
                mean_state = pt.zeros_like(simulated_object.state[0,0,:])
                mean_obs = pt.ones_like(simulated_object.observations[0,0,:])
                sd_state = pt.ones_like(simulated_object.state[0,0,:])
                sd_obs = pt.ones_like(simulated_object.observations[0,0,:])
            opt.zero_grad()
            complete_loss.clear_data()
            if normalise:
                simulated_object.state -= mean_state
                simulated_object.observations -= mean_obs 
                simulated_object.state /= sd_state
                simulated_object.observations /= sd_obs
            
            try:
                DPF.model.set_x_scaling(mean_state[0], sd_state[0])
            except:
                pass

            if lam != 0:
                likelihood_loss.clear_data()
                DPF_redefined.model.set_up(simulated_object.state[:,:,0:1], simulated_object.observations)
                DPF_redefined(simulated_object, T, likelihood_loss.get_reporters())
                likelihood_loss()
                complete_loss.register_data(weights=pt.tensor([1., lam], device='cuda'))
            loss.register_data(truth=simulated_object)
            
            
            DPF(simulated_object, T, complete_loss.get_reporters())
           
            complete_loss()
            complete_loss.backward()
            pt.nn.utils.clip_grad_value_(DPF.parameters(), clip)
            opt.step()

            train_loss[b + len(train)*epoch] = loss.item() * ((sd_state[0])**2)  

        times[epoch] = time.time() - start_ep

        if opt_schedule is not None:
            opt_schedule.step()
        DPF.eval()
        
        with pt.inference_mode():
            for simulated_object in valid:
                simulated_object.state -= mean_state
                simulated_object.observations -= mean_obs 
                simulated_object.state /= sd_state
                simulated_object.observations /= sd_obs 
                loss.clear_data()
                loss.register_data(truth=simulated_object)
                DPF(simulated_object, T, loss.get_reporters())
                test_loss[epoch] += loss().item() * ((sd_state[0])**2)

        if test_loss[epoch].item() < min_valid_loss:
            min_valid_loss = test_loss[epoch].item()
            best_dict = deepcopy(DPF.state_dict())

        if verbose:
            print(f'Epoch {epoch}:')
            print(f'Train loss: {np.mean(train_loss[epoch*len(train):(epoch+1)*len(train)])}')
            print(f'Validation loss: {test_loss[epoch]}\n')

    DPF.load_state_dict(best_dict)
    DPF.n_particles *= test_scaling
    DPF.ESS_threshold *= test_scaling

    start_test = time.time()
    results_ = _test(DPF, loss, T, test, (mean_state, sd_state, mean_obs, sd_obs))
    return *results_, times, np.array([time.time() - start_test])


def train_s2s(NN: pt.nn.Module, opt: pt.optim.Optimizer, data: pt.utils.data.Dataset, batch_size: Iterable[int], 
        set_fractions: Iterable[float], 
        epochs: int,
        opt_schedule: pt.optim.lr_scheduler.LRScheduler=None,
        verbose:bool=True,
        clip:float = pt.inf
        ):
    try:
        train_set, valid_set, test_set = pt.utils.data.random_split(data, set_fractions)
        if batch_size[0] == -1:
            batch_size[0] = len(train_set)
        if batch_size[1] == -1:
            batch_size[1] = len(valid_set)
        if batch_size[2] == -1:
            batch_size[2] = len(test_set)

        train = pt.utils.data.DataLoader(train_set, batch_size[0], shuffle=True, collate_fn=data.collate, num_workers= data.workers)
        valid = pt.utils.data.DataLoader(valid_set, min(batch_size[1], len(valid_set)), shuffle=False, collate_fn=data.collate, num_workers= data.workers)
        test = pt.utils.data.DataLoader(test_set, min(batch_size[2], len(test_set)), shuffle=False, collate_fn=data.collate, num_workers= data.workers, drop_last=True)
    except:
        train, valid, test = set_fractions
    train_loss = np.zeros(len(train)*epochs)
    test_loss = np.zeros(epochs)
    min_valid_loss = pt.inf
    times = np.empty(epochs)
    for epoch in range(epochs):
        start = time.time()
        NN.train()
        train_it = enumerate(train)
        for b, simulated_object in train_it:
            opt.zero_grad()
            x = NN(simulated_object.observations)
            loss = pt.mean((x - simulated_object.state[:, :, 0:1])**2)
            loss.backward()
            pt.nn.utils.clip_grad_value_(NN.parameters(), clip)
            opt.step()
            train_loss[b + len(train)*epoch] = loss.item()
        if opt_schedule is not None:
            opt_schedule.step()
        NN.eval()
        for simulated_object in valid:
            x = NN(simulated_object.observations)
            loss = pt.mean((x - simulated_object.state[:, :, 0:1])**2)
            test_loss[epoch] += loss.item()
        test_loss[epoch] /= len(valid)

        if test_loss[epoch] < min_valid_loss:
            min_valid_loss = test_loss[epoch]
            best_dict = deepcopy(NN.state_dict())

        if verbose:
            print(f'Epoch {epoch}:')
            print(f'Train loss: {np.mean(train_loss[epoch*len(train):(epoch+1)*len(train)])}')
            print(f'Validation loss: {test_loss[epoch]}\n')
        times[epoch] = time.time() - start
    NN.load_state_dict(best_dict)
    start_test = time.time()
    for simulated_object in test:
        x = NN(simulated_object.observations)
        loss = pt.mean((x - simulated_object.state[:, :, 0:1])**2, dim=2)
        loss = loss.to(device ='cpu').detach().numpy()
        print(f'Test loss: {np.mean(loss)}')
        return np.array([np.mean(loss)]), np.mean(loss, axis = 0), times, np.array([time.time() - start_test])
