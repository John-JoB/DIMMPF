from .model import (
    Feynman_Kac,
    Simulated_Object,
    HMM
)
from .utils import normalise_log_quantity
import numpy as np
from typing import Any, Callable, Union, Iterable
from copy import copy
from matplotlib import pyplot as plt
import torch as pt
from .resampling import Resampler
from torch import nn
import torch.autograd.profiler as profiler
from warnings import warn
#from .results import Reporter

"""

Main simulation functions for general non-differentiable particle filtering based on 
the algorithms presented in 'Choin, Papaspiliopoulos: An Introduction to Sequential 
Monte Carlo'

"""

 
class Differentiable_Particle_Filter(nn.Module):
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
        device: str = 'cuda'
        #state_scaling:float = 1., 
        #weight_scaling:float = 1.,
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

    def __copy__(self):
        return Differentiable_Particle_Filter(
            copy(self.model),
            copy(self.truth),
            self.n_particles,
            self.resampler,
            self.ESS_threshold,
        )
    
    def initialise(self, truth:Simulated_Object) -> None:
        
        
        self.t = 0
        
        self.truth = truth
        self.model.set_observations(self.truth._get_observation, 0)
        self.x_t = self.model.M_0_proposal(self.truth.state.size(0), self.n_particles)
        self.x_t.requires_grad = True
        if self.model.alg == self.model.PF_Type.Bootstrap:
            self.log_weights = self.model.log_f_t(self.x_t, 0)
        elif self.model.alg == self.model.PF_Type.Guided:
            self.log_weights = self.model.log_G_0_guided(self.x_t)
        else:
            self.log_weights = self.model.log_G_0(self.x_t)
        self.log_normalised_weights = normalise_log_quantity(self.log_weights)
        self.order = pt.arange(self.n_particles, device=self.device)
        self.resampled = True
        self.resampled_weights = pt.zeros_like(self.log_weights) - np.log(self.n_particles)
    

    def advance_one(self) -> None:
        """
        A function to perform the generic particle filtering loop (algorithm 10.3),
        advances the filter a single timestep.
        
        """

        self.t += 1
        if self.ESS_threshold < self.n_particles:
            mask = (1. / pt.sum(pt.exp(2*self.log_normalised_weights), dim=1)) < self.ESS_threshold
            resampled_x, resampled_w, _ = self.resampler(self.x_t[mask], self.log_normalised_weights[mask])
            self.x_t = self.x_t.clone()
            self.log_weights = self.log_normalised_weights.clone()
            self.x_t[mask] = resampled_x
            self.log_weights[mask] = resampled_w
            self.resampled = False
        else:
            self.x_t, self.log_weights, self.resampled_indices = self.resampler(self.n_particles, self.x_t, self.log_normalised_weights)
            self.resampled = True
        self.resampled_weights = self.log_weights.clone()
        self.x_t_1 = self.x_t.clone()
        self.model.set_observations(self.truth._get_observation, self.t)
        self.x_t = self.model.M_t_proposal(self.x_t_1, self.t)
        if self.model.alg == self.model.PF_Type.Bootstrap:
            self.log_weights += self.model.log_f_t(self.x_t, self.t)
        elif self.model.alg == self.model.PF_Type.Guided:
            self.log_weights += self.model.log_G_t_guided(self.x_t, self.x_t_1, self.t)
        else:
            self.log_weights += self.model.log_G_t(self.x_t, self.x_t_1, self.t)
        self.log_normalised_weights = normalise_log_quantity(self.log_weights)

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
    

    def display_particles(self, iterations: int, dimensions_to_show: Iterable, dims: Iterable[str], title:str):
        """
        Run the particle filter plotting particle locations in either one or two axes
        for each timestep. First plot after timestep one, each plot shows the current
        particles in orange, the previous particles in blue and, if availiable, the 
        true location of the observation generating object in red.

        Parameters
        ----------
        iterations: int
            The number of timesteps to run for

        dimensions_to_show: Iterable of int 
            Either length one or two, gives the dimensions of the particle state vector
            to plot. If length is one then all particles are plot at y=0.
        
        """
        if self.training:
            raise RuntimeError('Cannot plot particle filter in training mode please use eval mode')
        
        
        for i in range(iterations):
            x_last = self.x_t.clone()
            weights_last = self.log_normalised_weights.clone()
            self.advance_one()
            if len(self.x_t.shape) == 1:
                plt.scatter(x_last, np.zeros_like(self.x_t), marker="x")
                plt.scatter(self.x_t, np.zeros_like(self.x_t), marker="x")
                try:
                    print(self.truth.x_t)
                    print(self.model.y[self.t])
                    plt.scatter([self.truth.state[i+1]].detach(), [0], c="r")
                except AttributeError:
                    pass
                plt.show(block=True)
            elif len(dimensions_to_show) == 1:
                plt.scatter(
                    x_last[:, dimensions_to_show[0]].detach().to(device='cpu'),
                    np.zeros(len(self.x_t)),
                    marker="x",
                )
                plt.scatter(
                    self.x_t[:, dimensions_to_show[0]].detach().to(device='cpu'),
                    np.zeros(len(self.x_t)),
                    marker="x",
                    alpha=pt.exp(self.log_normalised_weights).detach().to(device='cpu')
                )
                try:
                    plt.scatter(self.truth.state[i+1, dimensions_to_show[0]].detach(), 0, c="r")
                except AttributeError:
                    pass
                plt.show(block=True)
            else:
                alpha = pt.exp(weights_last - pt.max(weights_last)).detach().to(device='cpu')
                plt.scatter(
                    x_last[0, :, dimensions_to_show[0]].detach().to(device='cpu'),
                    x_last[0, :, dimensions_to_show[1]].detach().to(device='cpu'),
                    marker="x", 
                    alpha=alpha
                )
                alpha = pt.exp(self.log_normalised_weights - pt.max(self.log_normalised_weights)).detach().to(device='cpu')
                plt.scatter(
                    self.x_t[0, :, dimensions_to_show[0]].detach().to(device='cpu'),
                    self.x_t[0, :, dimensions_to_show[1]].detach().to(device='cpu'),
                    marker="x",
                    alpha=alpha
                )
                plt.legend(['Current timestep particles', 'Previous timestep particles'])
                av = pt.sum(pt.exp(self.log_normalised_weights).unsqueeze(2)*self.x_t, dim=1).detach().cpu().numpy()
                
                try:
                    plt.scatter(
                        self.truth.state[0, i+1, dimensions_to_show[0]].detach().to(device='cpu'),
                        self.truth.state[0, i+1, dimensions_to_show[1]].detach().to(device='cpu'),
                        c="r",
                    )
                    plt.legend(['Previous timestep particles', 'Current timestep particles',  'Current timestep ground truth'])
                except AttributeError:
                    pass
                plt.scatter(av[0, dimensions_to_show[0]], av[0, dimensions_to_show[1]], c="g")
                plt.title(f'{title}: Timestep {i+1}')
                plt.xlabel(dims[0])
                plt.ylabel(dims[1])

                plt.show(block=True)

class HMM_Inference(nn.Module):
    '''
    Requires that the HMM has a bounded number of states
    '''

    def __init__(self, model: HMM, device: str ='cuda'):
        self.device = device
        self.model = model
        if not model.alg == model.PF_Type.Bootstrap:
            raise TypeError('The given HMM should be bootstrap')
        
    def __copy__(self):
        return HMM_Inference(copy(self.model), self.device)

    def initialise(self, truth:Simulated_Object):
        self.truth = truth
        self.states = self.model.generate_state_0().unsqueeze(0)
        self.prediction = self.model.log_M_0(self.states).squeeze()
        marginal_likelihoods = self.model.log_f_t(self.states, 0).squeeze()
        unnorm_filter = self.prediction + marginal_likelihoods
        self.likelihood_factor = pt.logsumexp(unnorm_filter)
        self.filter = unnorm_filter - self.likelihood_factor
        self.t = 0

    def advance_one(self):
        self.t += 1
        self.states_1 = self.states.clone() # 1xRxC
        self.states = self.model.generate_state_t(self.states_1, self.t) # 1xSxD
        repeat_states = pt.transpose(self.states, 0, 1).expand(self.states_1.size(1) , dim = 1) # SxRxD 
        repeat_states_1 = self.states_1.expand(self.states.size(0), dim=0) # SxRxC
        trans_probs =  self.model.log_M_t(repeat_states, repeat_states_1, self.t) # SxR
        self.prediction = pt.logsumexp(self.filter.unsqueeze(0).expand(self.states.size(1), 0) + trans_probs, dim = 1) # S
        marginal_likelihoods = self.model.log_f_t(self.states, 0).squeeze()
        unnorm_filter = self.prediction + marginal_likelihoods
        self.likelihood_factor = pt.logsumexp(unnorm_filter)
        self.filter = unnorm_filter - self.likelihood_factor
        
    def forward(self, T:int, truth:Simulated_Object):
        self.initialise(truth)
        likelihood = pt.tensor(0)
        likelihood = likelihood + self.likelihood_factor
        for i in range(T):
            self.advance_one()
            likelihood = likelihood + self.likelihood_factor
        return likelihood