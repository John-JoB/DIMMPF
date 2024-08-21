from .simulation import Differentiable_Particle_Filter
from .utils import normalise_log_quantity
import numpy as np
from matplotlib import pyplot as plt
#from joblib import cpu_count
from .model import SSM
import torch as pt
from typing import Callable


class Reporter():
    """
    Base class to report on a particle filter state,
    the copy function is modified to set the result vector to None to avoid
    overwriting it in using the copy.

    Parameters
    ----------
    **kwargs: Any
        The reporter parameters to be defined in implementation

    Implementation Notes
    -----------
    You should only use __init__ to set constants,
    any state that requires access to the particle filter to set-up or can change
    after initialisation should be initialised via the initialise method.

    Any subclass that overrides __init__() should call super().__init__().

    The initialise() method is used to set up initial values and memory and has
    access to the particle filter the Reporter should use. It should always be called
    before evaluate() is called for the first time. It first call super.initialise and
    at least should initialise self.results as an ndarray with self.expected_length
    entries along axis 0.

    evaluate() takes a particle filter (at time t) and should update entry t of
    self.results.

    """

    def __init__(self):
        self.results = None

    def __copy__(self):
        cls = self.__class__
        out = cls.__new__(cls)
        out.__dict__.update(self.__dict__)
        out.results = None
        return out

    def initialise(self, PF: Differentiable_Particle_Filter, iterations: int):
        self.expected_length = iterations + 1

    def evaluate(self, PF: Differentiable_Particle_Filter):
        raise NotImplementedError('Evaluate method not implemented')

    def plot(self, fun: Callable = lambda x: x):
        plt.plot(fun(self.results))

    def finalise(self, PF: Differentiable_Particle_Filter):
        pass


class Q_t(Reporter):
    """
    Reporter for the FK filtering mean under measures Q
    
    """

    def __init__(self, function_to_average):
        super().__init__()
        self.function_to_average = function_to_average

    def initialise(self, PF: Differentiable_Particle_Filter, iterations: int):
        super().initialise(PF, iterations)
        self.results = pt.empty(
            (self.expected_length, self.function_to_average(PF.x_t).size(1)),
            device=PF.device
        )

    def evaluate(self, PF: Differentiable_Particle_Filter):
        self.results[PF.t] = pt.einsum(
            "ji, j", self.function_to_average(PF.x_t), PF.normalised_weights
        )


class Q_t_1(Reporter):
    """
    Reporter for the FK predictive mean under measures Q
    
    """

    def __init__(self, function_to_average):
        super().__init__()
        self.function_to_average = function_to_average

    def initialise(self, PF: Differentiable_Particle_Filter, iterations: int):
        super().initialise(PF, iterations)
        self.results = pt.empty(
            (self.expected_length, self.function_to_average(PF.x_t).size(1)),
            device=PF.device
        )

    def evaluate(self, PF: Differentiable_Particle_Filter):
        weights = normalise_log_quantity(PF.log_weights_1)
        self.results[PF.t] = pt.einsum(
            "ji, j", self.function_to_average(PF.x_t), weights
        )


class log_l_t(Reporter):
    """
    Reporter for the FK log likelihood factors
    
    """
    def initialise(self, PF: Differentiable_Particle_Filter, iterations: int):
        super().initialise(PF, iterations)
        self.results = pt.empty(self.expected_length, device=PF.device)

    def evaluate(self, PF: Differentiable_Particle_Filter):
        self.results[PF.t] = pt.logsumexp(PF.log_weights) - pt.logsumexp(
            PF.log_weights_1
        )


class log_L_t(Reporter):
    """
    Reporter for the FK likelihood factors
    
    """

    def initialise(self, PF: Differentiable_Particle_Filter, iterations: int):
        super().initialise(PF, iterations)
        self.results = pt.empty(self.expected_length, device=PF.device)

    def evaluate(self, PF: Differentiable_Particle_Filter):
        if PF.t == 0:
            self.results[PF.t] = pt.logsumexp(PF.log_weights)
            return
        self.results[PF.t] = (
            pt.logsumexp(PF.log_weights)
            - pt.logsumexp(PF.log_weights_1)
            + self.results[PF.t - 1]
        )


class Filtering_Mean(Reporter):
    """
    Reporter for the state space filtering mean, equivelent to Q_t for the guided and
    bootstrap filters
    
    """

    def __init__(self, function_to_average):
        super().__init__()
        self.function_to_average = function_to_average

    def initialise(self, PF: Differentiable_Particle_Filter, iterations: int):
        super().initialise(PF, iterations)
        if not isinstance(PF.model, SSM):
            raise TypeError("Model must inherit from the SSM class")
        self.results = pt.empty((PF.x_t.size(0), self.expected_length, self.function_to_average(PF.x_t).size(2)), device=PF.device)

    def evaluate(self, PF: Differentiable_Particle_Filter):
        if PF.model.alg == PF.model.PF_Type.Auxiliary:
            if PF.t == 0:
                importance_weights = PF.model.log_G_0_guided(PF.x_t, PF.n_particles)
            else:
                importance_weights = (
                    PF.log_weights_1
                    + PF.model.log_G_t_guided(PF.x_t, PF.x_t_1, PF.t)
                    - PF.model.log_eta_t(PF.x_t_1, PF.t - 1)
                )
            norm_weights = normalise_log_quantity(importance_weights)
        else:
            norm_weights = pt.exp(PF.log_normalised_weights)
        
        self.results[:, PF.t, :] = pt.einsum("bji,bj->bi", self.function_to_average(PF.x_t), norm_weights)
        #self.results[PF.t] = pt.einsum("bji,bj->bi", self.function_to_average(PF.x_t), norm_weights)


class Predictive_Mean(Reporter):
    """
    Reporter for the state space predictive mean, equivelent to Q_t_1 for the
    bootstrap filter

    Parameters
    -----------
    function_to_average: Callable, default: Identity
        The function to apply to the particles before averaging

    forward: bool, default: False
        When true this modifies the previous weights for computation,
        when false it modifies the current weights. It will depend on the 
        specific model which is faster. No effect for bootstrap filtering,
        where forward computation is forced since it will always be the more
        efficient.

    Notes
    -----
    This proceedure uses future information to refine it's prediction so is unsuitable for the online case.

    This estimator converges in the case of a high number of particles, however in the
    regime of highly informative data, the target and proposal in the importance 
    sampling step are quite different and this estimator yields very unreliable results.
    
    """

    def __init__(self, function_to_average:Callable= lambda x: x, forward:bool = False):
        super().__init__()
        self.function_to_average = function_to_average
        self.forward = forward

    def initialise(self, PF: Differentiable_Particle_Filter, iterations: int):
        super().initialise(PF, iterations)
        if not isinstance(PF.model, SSM):
            raise TypeError("Model must inherit from the SSM class")
        self.results = pt.empty((PF.x_t.size(0), self.expected_length, self.function_to_average(PF.x_t).size(1)), device=PF.device)
        

    def _get_weights(self, PF: Differentiable_Particle_Filter):

        if PF.model.alg == PF.model.PF_Type.Bootstrap:
            return PF.resampled_weights
        
        if self.forward:
            log_importance_weights = PF.resampled_weights
            if PF.t == 0:
                return log_importance_weights - PF.model.log_R_0(PF.x_t)
            
            if PF.model.alg == PF.model.PF_Type.Auxiliary:
                log_importance_weights = log_importance_weights - PF.model.log_eta_t(PF.x_t_1, PF.t - 1)
            return log_importance_weights + PF.model.log_R_t(PF.x_t, PF.x_t_1, PF.t)
               
        log_importance_weights = PF.log_normalised_weights
        if PF.model.alg == PF.model.PF_Type.Auxiliary:
            log_importance_weights = log_importance_weights - PF.model.log_eta_t(PF.x_t, PF.t)
        log_importance_weights = log_importance_weights - PF.model.log_f_t(PF.x_t, PF.t)
        return log_importance_weights
    
    def evaluate(self, PF: Differentiable_Particle_Filter) -> None:
        log_importance_weights = self._get_weights(PF)
        log_importance_weights = normalise_log_quantity(log_importance_weights)
        self.results[:, PF.t, :] = pt.einsum("bji,bj->bi", self.function_to_average(PF.x_t), pt.exp(log_importance_weights))


class True_Predictive_Mean(Reporter):
    """
    Reporter for the state space predictive mean, equivelent to Q_t_1 for the
    bootstrap filter

    Parameters
    -----------
    function_to_average: Callable, default: Identity
        The function to apply to the particles before averaging

    Notes
    -----
    For true predictive estimation, this proceedure does not use future information to
    refine it's prediction so is suitable for the online case when delayed_computation is False.
    As such (in the non-bootstrap case) the model must additionally implement the dynamic model 
    as P_0_proposal() and P_t_proposal(). Equivelent to Predictive_Mean in the Bootstrap case. 
    """

    def __init__(self, function_to_average:Callable= lambda x: x):
        super().__init__()
        self.function_to_average = function_to_average

    def initialise(self, PF: Differentiable_Particle_Filter, iterations: int) -> None:
        super().initialise(PF, iterations)
        if not isinstance(PF.model, SSM):
            raise TypeError("Model must inherit from the SSM class")
        self.results = pt.empty((PF.x_t.size(0), self.expected_length, self.function_to_average(PF.x_t).size(1)), device=PF.device)
    

    def _get_state_and_weights(self, PF:Differentiable_Particle_Filter):
        log_importance_weights = PF.resampled_weights
        if PF.model.alg == PF.model.PF_Type.Bootstrap:
            return PF.x_t, log_importance_weights
        
        if PF.t == 0:
            pred_x_t = PF.model.P_0_proposal(PF.x_t.size(0), PF.x_t.size(1))
            return pred_x_t, log_importance_weights
        
        if PF.model.alg == PF.model.PF_Type.Auxiliary:
            log_importance_weights = log_importance_weights - PF.model.log_eta_t(PF.x_t_1, PF.t - 1)
        pred_x_t = PF.model.P_t_proposal(PF.x_t_1, PF.t)
        return pred_x_t, log_importance_weights

    def evaluate(self, PF: Differentiable_Particle_Filter) -> None:
        pred_x_t, log_importance_weights = self._get_state_and_weights(PF)
        log_importance_weights = normalise_log_quantity(log_importance_weights)
        self.results[:, PF.t, :] = pt.einsum("bji,bj->bi", self.function_to_average(pred_x_t), pt.exp(log_importance_weights))


class Log_Likelihood(Reporter):
    """
    Reporter for the state space Likelihood, equivelent to Log_L_t for the
    guided and bootstrap filters
    
    """
    def __init__(self):
        super().__init__()

    def initialise(self, PF: Differentiable_Particle_Filter, iterations: int):
        super().initialise(PF, iterations)
        if not isinstance(PF.model, SSM):
            raise TypeError("Model must inherit from the Auxiliary_Feynman_Kac class")
        self.results = pt.empty((PF.x_t.size(0), self.expected_length, 1), device=PF.device)
        self.FK_likelihood = pt.zeros(PF.x_t.size(0), device=PF.device)
        self.log_particles = np.log(PF.n_particles)

    def evaluate(self, PF: Differentiable_Particle_Filter):

        #sum_resampled_weights =  pt.logsumexp(PF.resampled_weights, dim=1)
        sum_of_weights = PF.log_weights[:, 0] - PF.log_normalised_weights[:, 0] - self.log_particles #sum_resampled_weights
        
        if PF.model.alg == PF.model.PF_Type.Auxiliary:
            if PF.t == 0:
                importance_weights = PF.model.log_G_0_guided(PF.x_t, PF.n_particles)
            else:
                importance_weights = PF.resampled_weights + PF.model.log_G_t_guided(PF.x_t, PF.x_t_1, PF.t) - PF.model.log_eta_t(PF.x_t_1, PF.t - 1)
            self.results[:, PF.t, :] = pt.logsumexp(importance_weights, dim = 1) + self.FK_likelihood
            self.FK_likelihood = self.FK_likelihood + sum_of_weights
            return
        self.FK_likelihood = self.FK_likelihood + sum_of_weights
        self.results[:, PF.t, :] = self.FK_likelihood.unsqueeze(1)


class Log_Likelihood_Factors(Reporter):
    """
    Reporter for the state space log Likelihood factors, equivelent to Log_l_t
    for the guided and bootstrap filters
    
    """

    def initialise(self, PF: Differentiable_Particle_Filter, iterations: int):
        super().initialise(PF, iterations)
        if not isinstance(PF.model, SSM):
            raise TypeError("Model must inherit from the Auxiliary_Feynman_Kac class")
        self.results = pt.empty((PF.x_t.size(0), self.expected_length, 1), device=PF.device)
        self.likelihood = Log_Likelihood()
        self.likelihood.initialise(PF, iterations)

    def evaluate(self, PF: Differentiable_Particle_Filter):
        self.likelihood.evaluate(PF)
        if PF.t == 0:
            self.results[:, 0, :] = self.likelihood.results[:, 0, :]
            return

        self.results[:, PF.t, :] = self.likelihood.results[:, PF.t, :] - self.likelihood.results[:, PF.t - 1, :]


class ESS(Reporter):
    """
    Reporter for the effective sum of squares
    
    """

    def initialise(self, PF: Differentiable_Particle_Filter, iterations: int):
        super().initialise(PF, iterations)
        self.results = pt.empty(self.expected_length, device=PF.device)

    def evaluate(self, PF: Differentiable_Particle_Filter):
        self.results[PF.t] = 1 / pt.sum(PF.normalised_weights**2)


class Survival_Ratio(Reporter):
    """
    Reporter for the survival ratio of particles over resampling
    
    """

    def initialise(self, PF: Differentiable_Particle_Filter, iterations: int):
        super().initialise(PF, iterations)
        self.results = pt.empty(self.expected_length, device=PF.device)

    def evaluate(self, PF: Differentiable_Particle_Filter):
        if PF.t == 0:
            self.results[PF.t] = 1
            return
        self.results[PF.t] = len(pt.unique(PF.resampled_indices)) / PF.n_particles

'''
def aggregate_many_runs(
    runs: int,
    PF: Vanilla_Particle_Filter,
    iterations: int,
    statistics,
    cores=cpu_cores,
    aggregation_functions=None,
    initialisation_functions=None,
    finialisation_functions=None,
):
    """
    Aggregate a number of runs of a particle filter on the same data, under a set of
    custom or provided aggregation schemes. Particle filters are run in parallel.

    Custom aggregations must have the form:
    F(A(A(A....A(I(r_1)r_2)....r_n-1)r_n))
    For functions F, A, and I and results r_i.

    Parameters
    -----------
    runs: int
        The number of times to run the particle filter

    PF: Vanilla_Particle_Filter
        The particle filter to run

    iterations: int
        The number of timesteps to run the particle filter for

    statistics: (S,) Ordered iterable of Reporter
        The statistics to be aggregated

    cores: int, optional, defaults to the number of availiable cores
        The number of cpu cores to run particle filters on. If this is more than are
        availiable then default to the maximum availiable

    aggregation_functions: (S,) Ordered iterable of (String or Callable), or None, optional, default None
        The functions to elementwise aggregate the results specified by statistics with according to their order,
        given options are:
            'mean': mean
            'sum': sum
            'max': maximum
            'min': minimum
            'var': variance
            'stdev': standard deviation
        Mean and sum can take vector outputs, but the other aggregations should only be used on scalar outputs
        Alternatively one may provide a funcion (T, d) ndarray * (T, d) ndarray -> (T, d) ndarray

        All string refered functions must come after all custom functions
        If None then the mean is returned for all statistics.

    intitialisation_functions: (Number of custom aggregation_functions,) Ordered iteratable of Callable or None
        The function to apply to the first recieved result. Should not be given if using a provided
        aggregation function. Ordered the same as the custom aggregation_functions. Can be None if all
        aggregation_functions are string defined.
        Provided functions should be (T, d) ndarray -> (T, d) ndarray

    finalisation_functions: (Number of custom aggregation_functions,) Ordered iteratable of Callable or None
        The function to apply to the final result of aggregation. Should not be given if using a provided
        aggregation function. Ordered the same as the custom aggregation_functions. Can be None if all
        aggregation_functions are string defined.
        Provided functions should be (T, d) ndarray -> (T, d) ndarray

    Returns
    --------
    aggregations: (S,) List of (T, d) ndarray
        A list of the the particle filter statistics aggregated according to the provided aggregation functions

    """  # noqa: E501

    if aggregation_functions is None:
        aggregation_functions = ["mean"] * len(statistics)

    if initialisation_functions is None:
        initialisation_functions = [None] * len(aggregation_functions)
        finialisation_functions = [None] * len(aggregation_functions)

    if len(initialisation_functions) < len(aggregation_functions):
        initialisation_functions += [None] * (
            len(aggregation_functions) - len(initialisation_functions)
        )
        finialisation_functions += [None] * (
            len(aggregation_functions) - len(finialisation_functions)
        )

    def identity(a):
        return a

    for i, agg in enumerate(aggregation_functions):
        if agg == "mean":
            aggregation_functions[i] = pt.add
            initialisation_functions[i] = identity
            finialisation_functions[i] = lambda a: a / runs
        if agg == "sum":
            aggregation_functions[i] = pt.add
            initialisation_functions[i] = identity
            finialisation_functions[i] = identity
        if agg == "max":
            aggregation_functions[i] = pt.maximum
            initialisation_functions[i] = identity
            finialisation_functions[i] = identity
        if agg == "min":
            aggregation_functions[i] = pt.minimum
            initialisation_functions[i] = identity
            finialisation_functions[i] = identity
        if agg == "var":
            aggregation_functions[i] = lambda a, b: pt.add(a, pt.vstack((b**2, b)))
            initialisation_functions[i] = lambda a: pt.vstack((a**2, a))
            finialisation_functions[i] = lambda a: (a[0] - (a[1] ** 2) / runs) / (
                runs - 1
            )
        if agg == "stdev":
            aggregation_functions[i] = lambda a, b: pt.add(a, pt.vstack((b**2, b)))
            initialisation_functions[i] = lambda a: pt.vstack((a**2, a))
            finialisation_functions[i] = lambda a: pt.sqrt(
                (a[0] - (a[1] ** 2) / runs) / (runs - 1)
            )

    def run_helper():
        statistics_copy = [copy(stat) for stat in statistics]
        copy(PF).run(iterations, statistics_copy)
        return [stat.report_results() for stat in statistics_copy]

   results = parallelise((run_helper for _ in range(runs)), cores)

    first_result = next(results)
    aggregations = [None] * len(statistics)
    for i in range(len(aggregations)):
        aggregations[i] = initialisation_functions[i](first_result[i])

    for res in results:
        for i in range(len(aggregations)):
            aggregations[i] = aggregation_functions[i](aggregations[i], res[i])

    for i in range(len(aggregations)):
        aggregations[i] = finialisation_functions[i](aggregations[i])

    return aggregations
'''