from dpf_rs.model import (
    Feynman_Kac,
    Simulated_Object,
    HMM
)
from dpf_rs.utils import normalise_log_quantity
import numpy as np
from typing import Any, Callable, Union, Iterable
from copy import copy
from matplotlib import pyplot as plt
import torch as pt
from dpf_rs.resampling import Resampler
from torch import nn
import torch.autograd.profiler as profiler
from warnings import warn
from dpf.resampling import batched_reindex
#from .results import Reporter

"""

Main simulation functions for general non-differentiable particle filtering based on 
the algorithms presented in 'Choin, Papaspiliopoulos: An Introduction to Sequential 
Monte Carlo'

"""

 
class MADPF(nn.Module):
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
        merge_prob: int = 1 
    ) -> None:
        super().__init__()
        self.device = device
        self.resampler = resampler
        resampler.to(device=device)
        self.ESS_threshold = ESS_threshold
        self.n_particles = n_particles
        self.model = model
        self.merge_prob = merge_prob

        self.n_models = self.model.n_models
        if self.model.alg == self.model.PF_Type.Undefined:
            warn('Filtering algorithm not set')
        self.model.to(device=device)

    def __copy__(self):
        return MADPF(
            copy(self.model),
            copy(self.truth),
            self.n_particles,
            self.resampler,
            self.ESS_threshold,
        )
    
    def weight_models(self):
        index = self.x_t[:, :, 1].to(int)
        model_posteriors = pt.zeros(self.x_t.size[0], self.n_models)
        for m in range(self.n_models):
            mask = (index == m).to(int)
            m_1 = pt.sum(mask, dim=1)
            model_posteriors[:, m] =  pt.sum(pt.exp(self.log_normalised_weights)*mask, dim = 1)/m_1
        return model_posteriors/pt.sum(model_posteriors, dim = 1)


    def initialise(self, truth:Simulated_Object) -> None:
        
        
        self.t = 0
        
        self.truth = truth
        self.model.set_observations(self.truth._get_observation, 0)
        self.x_t = self.model.M_0_proposal(self.truth.state.size(0), self.n_particles)
        self.masks = [0]*self.n_models
        for m in range(self.n_models):
            self.masks[m] = (self.x_t[:, :, 1].to(int) == m)
        self.x_t.requires_grad = True
        self.log_weights = self.model.log_f_t(self.x_t, 0)
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
        

        if pt.all(pt.rand(1) < self.merge_prob) or self.t==1:
            self.x_t, self.log_weights, self.resampled_indices = self.resampler(self.x_t, self.log_normalised_weights)
            self.x_t[:, :, 1] = pt.multinomial(pt.ones(self.n_models), (self.x_t.size(0) * self.x_t.size(1)), replacement=True).reshape((self.x_t.size(0),self.x_t.size(1)))
            for m in range(self.n_models):
                self.masks[m] = (self.x_t[:, :, 1].to(int) == m)
        else:
            base_particles = self.n_particles//(self.n_models*5)
            samples_per_model = (base_particles + pt.floor((self.n_particles-base_particles*self.n_models)*pt.exp(self.model_posteriors))).to(int)
            missing_samples = self.n_particles - pt.sum(samples_per_model, dim= 1)
            samples_per_model[:, 0] = samples_per_model[:, 0] + missing_samples
            
            new_x_t = pt.empty_like(self.x_t, device=self.device)
            new_masks = [pt.zeros_like(self.masks[0]) for _ in range(self.n_models)]
            cum_index = pt.zeros(self.x_t.size(0), device=self.device, dtype=int)
            new_weights = pt.zeros_like(self.log_weights, device=self.device)
            for m in range(self.n_models):
                for b in range(self.x_t.size(0)):
                    new_masks[m][b, cum_index[b]:cum_index[b]+samples_per_model[b, m]] = 1
                cum_index += samples_per_model[:, m]
                new_masks[m] = new_masks[m].to(bool)
                temp_weights = pt.where(self.masks[m], self.modelwise_weights,  - pt.inf)
                resampled_p, _, _ = self.resampler(self.x_t, temp_weights)
                new_x_t[new_masks[m]] = resampled_p[new_masks[m]]
                new_weights = new_weights + self.cond_likelihoods[:, m:m+1] * new_masks[m].to(int)

            self.log_weights = new_weights
            self.x_t = new_x_t
            self.masks = new_masks


        self.resampled_weights = self.log_weights.clone()
        self.x_t_1 = self.x_t.clone()
        self.model.set_observations(self.truth._get_observation, self.t)
        self.x_t = self.model.M_t_proposal(self.x_t_1, self.t)
        self.log_weights += self.model.log_f_t(self.x_t, self.t)
        
        self.cond_likelihoods = pt.zeros(self.x_t.size(0), self.n_models, device=self.device)
        self.modelwise_weights = pt.empty_like(self.log_weights, device=self.device)
        for m in range(self.n_models):
            weights_temp = pt.where(self.masks[m], self.log_weights,  - pt.inf)
            
            m_1 = pt.sum(self.masks[m], dim=1)
            self.cond_likelihoods[:, m] =  pt.logsumexp(weights_temp, dim = 1) - pt.log(m_1)
            weights_temp = normalise_log_quantity(weights_temp)
            self.modelwise_weights = pt.where(self.masks[m], weights_temp, self.modelwise_weights)

        self.model_posteriors = normalise_log_quantity(self.cond_likelihoods)
    

        self.log_normalised_weights = pt.empty_like(self.log_normalised_weights, device=self.device)
        for m in range(self.n_models):
            self.log_normalised_weights[self.masks[m]] = (self.modelwise_weights + self.model_posteriors[:, m:m+1])[self.masks[m]]
        

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
        IMMtype: str = 'normal',
        grad_scale: float = 1.
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
        self.IMMtype = IMMtype
        self.grad_scale = pt.tensor(grad_scale, device=device)

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
        def forward(ctx:Any, input:pt.Tensor, grad_scale):
            ctx.save_for_backward(grad_scale)
            return input.clone()
        
        @staticmethod
        def backward(ctx, d_dinput):
            grad_scale = ctx.saved_tensors[0]
            return pt.clip(d_dinput,-10, 10), None
    

    def advance_one(self) -> None:
            """
            A function to perform the generic particle filtering loop (algorithm 10.3),
            advances the filter a single timestep.
            
            """

            self.t += 1
            regime_probs = self.model.get_regime_probs(self.x_t)
            adj_regime_probs = regime_probs + self.true_weights[:,:,None]
            adj_regime_probs = self.scale_grad.apply(adj_regime_probs, self.grad_scale)
            self.x_t = self.scale_grad.apply(self.x_t, self.grad_scale)
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
            #print(old_weights.size())
            #print(new_weights[0].size())
            if self.IMMtype == 'new':
                self.log_weights = pt.concat([self.model.log_f_t(k, self.x_t[k], self.t) + new_weights[k] + tot_regime_probs[:, None, k].detach() for k in range(self.Nk)], dim=1)
            elif self.IMMtype== 'OT':
                self.log_weights = pt.concat([self.model.log_f_t(k, self.x_t[k], self.t) + new_weights[k] + tot_regime_probs[:, None, k] for k in range(self.Nk)], dim=1)
            else:    
                self.log_weights = pt.concat([self.model.log_f_t(k, self.x_t[k], self.t) + new_weights[k] + old_weights[:, k*self.particles_per_model:(k+1)*self.particles_per_model, k] - old_weights[:, k*self.particles_per_model:(k+1)*self.particles_per_model, k].detach() + tot_regime_probs[:, None, k].detach() for k in range(self.Nk)], dim=1)
            self.x_t = pt.concat(self.x_t, dim=1)
            #self.x_t = pt.clip(self.x_t, -100, 100)
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
                #print(weights[0])
                self.log_weights = self.log_weights + weights - weights.detach()
                self.log_normalised_weights = normalise_log_quantity(self.log_weights)
                self.true_weights = self.log_normalised_weights

                        

            #   ind_dist = pt.stack([self.model.log_M_t(k, self.x_t, self.x_t_1, self.t) + old_weights[:, :, k] for k in range(self.Nk)], dim=2)
            #   likelihoods = pt.stack([self.model.log_f_t(k, self.x_t, self.t) for k in range(self.Nk)], dim=2)
            #    num = pt.logsumexp(ind_dist + likelihoods, dim=2)
            #    den = pt.logsumexp(ind_dist - tot_regime_probs[:, None, :], dim=2)
            #   self.log_normalised_weights = normalise_log_quantity(num-den)



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
