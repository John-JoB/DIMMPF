from dpf_rs.model import (
    Feynman_Kac,
    Simulated_Object,
)
from dpf_rs.utils import normalise_log_quantity
import numpy as np
from typing import Any, Union, Iterable
from copy import copy
from matplotlib import pyplot as plt
import torch as pt
from dpf_rs.resampling import Resampler
from torch import nn
from warnings import warn
from dpf_rs.resampling import batched_reindex


class IMM_Particle_Filter(nn.Module):
    """
    Class defines a particle filter for a generic Feynman-Kac model, the Bootstrap,
    Guided and Auxiliary formulations should all use the same algorithm

    On initiation the initial particles are drawn from M_0, to advance the model a
    timestep call the forward function

    Parameters
    ----------
    model: Feynmanc_Kac
        The model to perform particle filtering on

    truth: Simulated_Object
        The object that generates/reports the observations

    n_particles: int
        The number of particles to simulate

    resampler: func(X, (N,) ndarray) -> X
        This class imposes no datatype for the state, but a sensible definition of
        particle filtering would want it to be an iterable of shape (N,),
        a lot of the code that interfaces with this class assumes that it is an ndarray
        of floats.

    ESS_threshold: float or int
        The ESS to resample below, set to 0 (or lower) to never resample and n_particles
        (or higher) to resample at every step
    
    """

    def __init__(
        self,
        model: Feynman_Kac,
        n_particles: int,
        resampler: Resampler,
        ESS_threshold: Union[int, float], 
        device: str = 'cuda',
        IMMtype: str = 'normal'
    ) -> None:
        super().__init__()
        self.device = device
        self.resampler = resampler
        resampler.to(device=device)
        self.ESS_threshold = ESS_threshold
        self.n_particles = n_particles
        self.model = model
        if self.model.alg == self.model.PF_Type.Undefined:
            warn('Filtering algorithm not set')
        self.model.to(device=device)
        self.IMMtype = IMMtype

    def __copy__(self):
        return IMM_Particle_Filter(
            copy(self.model),
            copy(self.truth),
            self.n_particles,
            self.resampler,
            self.ESS_threshold,
        )
    
    def initialise(self, truth:Simulated_Object) -> None:
        
        
        self.t = 0
        self.truth = truth
        self.Nk = self.model.n_models
        self.model.set_observations(self.truth._get_observation, 0)
        self.particles_per_model = self.n_particles//self.Nk        
        self.x_t = pt.concat([self.model.M_0_proposal(k, self.truth.state.size(0), self.particles_per_model) for k in range(self.Nk)], dim =1)
        self.log_weights = pt.concat([self.model.log_f_t(k, self.x_t[:, k*self.particles_per_model:(k+1)*self.particles_per_model, :], self.t) for k in range(self.Nk)], dim=1)
        self.log_normalised_weights = normalise_log_quantity(self.log_weights)
        self.order = pt.arange(self.n_particles, device=self.device)
        self.resampled = True
        self.resampled_weights = pt.zeros_like(self.log_weights) - np.log(self.n_particles)
        self.true_weights = self.log_normalised_weights

    class scale_grad(pt.autograd.Function):
        @staticmethod
        def forward(ctx:Any, input:pt.Tensor):
            return input.clone()
        
        @staticmethod
        def backward(ctx, d_dinput):
            return pt.clip(d_dinput,-10, 10), None
    

    def advance_one(self) -> None:
            """
            A function to perform the generic particle filtering loop (algorithm 10.3),
            advances the filter a single timestep.
            
            """

            self.t += 1
            regime_probs = self.model.get_regime_probs(self.x_t)
            adj_regime_probs = regime_probs + self.true_weights[:,:,None]
            adj_regime_probs = self.scale_grad.apply(adj_regime_probs)
            self.x_t = self.scale_grad.apply(self.x_t)
            tot_regime_probs = pt.logsumexp(adj_regime_probs, dim=1)
            regime_resampling_weights = adj_regime_probs - tot_regime_probs[:, None, :]
            xs = [None]*self.Nk
            indices = [None]*self.Nk
            new_weights = [None]*self.Nk
            old_particles = self.x_t.clone()
            for k in range(self.Nk):
                xs[k], new_weights[k], indices[k] = self.resampler(self.particles_per_model, self.x_t.detach(), regime_resampling_weights[:, :, k].detach())
            self.x_t_1 = pt.concat(xs, dim=1)
            #if True:
            #    print(xs[0][0])
            if self.IMMtype == 'normal':
                indices = pt.concat(indices, dim=1)
                old_weights = batched_reindex(adj_regime_probs, indices)
            self.resampled = True
            self.resampled_weights = self.log_weights.clone()
            self.model.set_observations(self.truth._get_observation, self.t)
            self.x_t = [self.model.M_t_proposal(k, xs[k], self.t) for k in range(self.Nk)]
            if self.IMMtype == 'new':
                self.log_weights = pt.concat([self.model.log_f_t(k, self.x_t[k], self.t) + new_weights[k] + tot_regime_probs[:, None, k].detach() for k in range(self.Nk)], dim=1)
            elif self.IMMtype== 'OT':
                self.log_weights = pt.concat([self.model.log_f_t(k, self.x_t[k], self.t) + new_weights[k] + tot_regime_probs[:, None, k] for k in range(self.Nk)], dim=1)
            else:    
                self.log_weights = pt.concat([self.model.log_f_t(k, self.x_t[k], self.t) + new_weights[k] + old_weights[:, k*self.particles_per_model:(k+1)*self.particles_per_model, k] - old_weights[:, k*self.particles_per_model:(k+1)*self.particles_per_model, k].detach() + tot_regime_probs[:, None, k].detach() for k in range(self.Nk)], dim=1)
            self.x_t = pt.concat(self.x_t, dim=1)
            self.true_weights = normalise_log_quantity(self.log_weights)
            self.log_normalised_weights = self.true_weights
            if self.IMMtype == 'new' and self.training:
                weights = [None]*self.Nk
                for k in range(self.Nk):
                    start_range = k*self.particles_per_model
                    end_range = (k+1)*self.particles_per_model
                    weights[k] = adj_regime_probs[:, None, :, k] + self.model.log_M_t(k, self.x_t[:, start_range:end_range], old_particles, self.t)
                weights = pt.concat(weights, dim=1)
                weights = pt.logsumexp(weights, dim=2)
                self.log_weights = self.log_weights + weights - weights.detach()
                self.log_normalised_weights = normalise_log_quantity(self.log_weights)
                self.true_weights = self.log_normalised_weights


    def forward(self, sim_object: Simulated_Object, iterations: int, statistics: Iterable):

        """
        Run the particle filter for a given number of time step
        collating a number of statistics

        Parameters
        ----------
        iterations: int
            The number of timesteps to run for

        statistics: Sequence of result.Reporter
            The statistics to note during run results are stored
            in these result.Reporter objects
        """
        self.initialise(sim_object)

        for stat in statistics:
            stat.initialise(self, iterations)

        for _ in range(iterations + 1):
            for stat in statistics:
                stat.evaluate(PF=self)
            if self.t == iterations:
                break
            self.advance_one()
        
        stat.finalise(self)

        return statistics