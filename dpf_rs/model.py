from typing import Callable, Iterable, Generator
import copy
import torch as pt
import os
import shutil
from enum import Enum
from warnings import warn
from .utils import nd_select, batched_select

class Feynman_Kac(pt.nn.Module):

    """
    Abstract base class for a Feynman-Kac model with all the functions required for
    particle filtering needing implementation

    Any model for particle filtering should sub-class 'Feynman_Kac'

    M_0_proposal and M_t_proposal should sample from their respective distributions

    G_0, G_t are pointwise density evaluation functions

    For generality and notational consistency, the observations are treated like model
    parameters and are set at each time-step rather than passed as function parameters
    For convienence I provide a separate update function for them

    I allow for the model to take any number of parameters, that should be passed as
    keyword arguments to __init__, and may be updated at anytime via the
    set_model_parameters method

    I override the copy functionality to create a copy but with new rng.

    Get modules should return the all the neural networks in the model as a ModuleList

    Parameters
    ----------
    **kwargs: any
        The model parameters to be passed to set_model_parameters

    """

    class PF_Type(Enum):
        Undefined = 0
        Bootstrap = 1
        Guided = 2
        Auxiliary = 3

    class reindexed_array:
        """
        Inner class to reindex an array, should be taken as immutable and read-only
        after creation
        Intended usage is to access observations at time t as y[t]

        Parameters
        ----------
        base_index: int
            The desired index of the first stored item.

        *args: any
            Arguments passed to np.array() for array creation

        **kwargs: any
            Arguments passed to np.array() for array creation
            
        """

        def __init__(self, base_index: int, ls):
            super().__init__()
            self.array = ls
            self.base_index = base_index

        def __getitem__(self, index):
            return self.array[index - self.base_index]
        

    def set_observations(self, get_observation: Callable, t: int):
        NotImplementedError('Function to set observations not implemented')

    def to(self, **kwargs):
        if kwargs['device'] is not None:
            self.device = kwargs['device']
        for var in vars(self):
            if isinstance(var, pt.Tensor) and not isinstance(var, pt.nn.Parameter):
                var.to(dtype=kwargs['dtype'], device=kwargs['device'])
        super().to(**kwargs)
        

    def __init__(self, device:str='cuda') -> None:
        super().__init__()
        self.alg = self.PF_Type.Undefined
        self.device = device
        self.rng = pt.Generator(device=self.device)        

    # Evaluate G_0
    def log_G_0(self, x_0):
        NotImplementedError('Weighting function not implemented for time 0')

    # Sample M_0
    def M_0_proposal(self, batches:int, n_samples:int):
        NotImplementedError('Proposal model sampler not implemented for time 0')

    def log_M_0(self, x_0):
        NotImplementedError('Proposal density not implemented for time 0')

    # Evaluate G_t
    def log_G_t(self, x_t, x_t_1, t: int):
        NotImplementedError('Weighting function not implemented for time t')

    # Sample M_t
    def M_t_proposal(self, x_t_1, t: int):
        NotImplementedError('Proposal model sampler not implemented for time t')

    def log_M_t(self, x_t, x_t_1, t:int):
        NotImplementedError('Proposal density not implemented for time t')

    def observation_generation(self, x_t):
        raise NotImplementedError('Observation generation not implemented')


class SSM(Feynman_Kac):

    """
    Base class for an auxiliary Feynman-Kac model

    Notes
    ------
    R_t are the Raydon-Nikodym derivatives M_t(x_t-1, dx_t) / P_t(x_t-1, dx_t) and
    should be computable for standard useage

    I provide the standard form for calculating the auxiliary weight functions G_t
    but they can be overridden with a direct calculation if desired
    either for performance or that it is possible to particle filter if the
    M_t(x_t-1, dx_t) / P_t(x_t-1, dx_t) are not computable/divergent but the
    f(x_t)M_t(x_t-1, dx_t) / P_t(x_t-1, dx_t) are. In in which case you should not
    define the ratios R. Although it will not be possible
    to recover the predictive distribution through the usual importance
    sampler in this case.
    """

    def __init__(self, device: str = 'cuda') -> None:
        super().__init__(device)
        self.PF_type = 'Auxiliary'

    def log_R_0(self, x_0):
        raise NotImplementedError('Dynamic/Proposal Radon-Nikodym derivative not implemented for time zero')

    def log_R_t(self, x_t, x_t_1, t: int):
        raise NotImplementedError('Dynamic/Proposal Radon-Nikodym derivative not implemented for time t')

    def log_f_t(self, x_t, t: int):
        raise NotImplementedError('Observation likelihood not implemented')

    def log_eta_t(self, x_t, t: int):
        raise NotImplementedError('Auxililary weights not implemented')

    def log_G_0_guided(self, x_0):
        return self.log_R_0(x_0) + self.log_f_t(x_0, 0)

    def log_G_t_guided(self, x_t, x_t_1, t: int):
        return self.log_R_t(x_t, x_t_1, t) + self.log_f_t(x_t, t)

    def log_G_0(self, x_0):
        return self.log_G_0_guided(x_0) + self.log_eta_t(x_0, 0)

    def log_G_t(self, x_t, x_t_1, t: int):
        return (
            self.log_G_t_guided(x_t, x_t_1, t)
            + self.log_eta_t(x_t, t)
            - self.log_eta_t(x_t_1, t - 1)
        )

class HMM(SSM):

    def generate_state_0(self):
        raise NotImplementedError('State generation not implemented for time 0')

    def M_0_proposal(self, batches:int, n_samples: int):
        state = self.generate_state_0() #SxD
        probs = self.log_M_0(state.unsqueeze(0)).squeeze() #S
        indices = pt.multinomial(pt.exp(probs), batches*n_samples, True).reshape(batches, n_samples) #BxN
        return nd_select(state, indices)#BxNxD

    def generate_state_t(self, x_t_1, t:int):
        raise NotImplementedError('State generation not implemented for time t')

    def M_t_proposal(self, x_t_1, t: int):
        state = self.generate_state_t(x_t_1) #BxNxSxD
        probs = self.log_M_t(state, x_t_1, t) #BxNxS
        indices = pt.multinomial(pt.flatten(pt.exp(probs), 0,1), 1, True).reshape(x_t_1.shape(0), x_t_1.shape(1)) #BxN
        return batched_select(state, indices) #BxNxD


class State_Space_Object():
    """
    Base class for a generic state space object that can generate observations and
    update it's state

    The true state need not be availiable in which case this object can act like a
    queue of observations and return NaN whenever asked to evaluate the true state

    I keep a list that is assumed to contain observations at successive timesteps
    as well as an indexing variable to store the timestep of the first value in the
    list. Each time an observation is required, if it exists return it, if not
    advance the state and add observations sequentially to an array until the
    desired time is reached.

    I also provide copy functionality so that
    a copied state space object is a new state space object with the same RNG seeds
    so that running it again will produce consistent results

    Parameters
    -----------
    observation_history_length: int
        The number of observations to keep at any timestep

    observation_dimension: int
        The dimension of the observation vector

    Notes
    --------
    It is not recomended to interface with the class outside creation and the
    get_observation() method.

    The rng for transitioning states and for returning observations are separate
    because they can be done in different orders e.g. states 1:N can be generated
    without accessing any observations, but both states and observations are
    sequentially generated within their own series. Obviously do not try to access
    the rng generators outside of a child class. If you have time varying
    components to a subclass, the subclass must also override __copy__ and reset the
    relavent class state.

    """

    def _get_observation(self, t):
        pass
        
    def save(self):
        pass


class Simulated_Object(State_Space_Object):

    """Base class for a simulated object. This object simulates a hidden Markov process,
    for an interpretable output the given model should always be Bootstrap
    regardless of what algorithm is going to be used to filter. Past states are not stored
    so if the states are wanted then record it after every call to _get_observation. Filtering directly on
    a Simulated_Object is possible but not recomended. Future updates may improve functionality but as of now the 
    only reason to do this would be to run a filter indefinately, this has limited use in the DPF case. 

    Parameters
    ----------
    Model: Feynman_Kac
        The model to simulate, must be Bootstrap

    Batches: int
        The number of trajectories to simulate in paralel
    
    observation_history_length: int
        The minimum number of observations to keep stored in memory

    observation_dimension: int
        The number of dimensions the observations will have

    

    """

    def __copy__(self):
        cls = self.__class__
        out = cls.__new__(cls)
        out.__dict__.update(self.__dict__)
        out.observations = pt.empty_like(out.observations, device=self.device)
        out.time_index = 0
        out.object_time = 0
        out.first_object_set = False
        out.model = copy.copy(out.model)
        out.x_t = out.model.M_0_proposal(out.batch_size, 1)
        return out

    def __init__(self, model: Feynman_Kac, batch_size: int, observation_history_length: int, observation_dimension: int, device:str = 'cuda'):
        self.device = device
        self.observation_history_length = observation_history_length
        self.observation_dimension = observation_dimension
        self.observations = pt.empty((batch_size, self.observation_history_length * 2, self.observation_dimension), device=self.device)
        self.first_object_set = False
        self.time_index = 0
        self.object_time = 0
        self.model = model
        self.batch_size = batch_size
        self.x_t = self.model.M_0_proposal(batch_size, 1)   

    def _forward(self):
        self.object_time += 1
        self.x_t = self.model.M_t_proposal(self.x_t, self.object_time)

    def _set_observation(self, t: int, value: pt.Tensor) -> None:
        """

        Update observation history with a new observation, if the observation is
        history is full copy the second half of the values stored to the first half of
        the array and start filling from the half way point

        Parameters
        ----------
            t: int
                Timestep of new observation

            value: ndarray
                Value of new observation

        """
        if self.time_index + self.observation_history_length * 2 <= t:
            self.observations[:, :self.observation_history_length] = self.observations[:, self.observation_history_length:]
            self.time_index += self.observation_history_length
        self.observations[:, t - self.time_index, :] = value.squeeze(1)

    def _get_observation(self, t):
        """
        Fetch observation at time t if it is not created then advance the object
        state and generate observations until time t
        """
        if t < 0:
            return pt.full((self.batch_size, self.observation_dimension), pt.nan, device=self.device)
        
        if t < self.time_index:
            raise ValueError(
                f"Trying to access observation at time {t}, "
                f"the earliest stored is at time {self.time_index}"
            )

        if t == 0 and not self.first_object_set:
            self.first_object_set = True
            self._set_observation(0, self.model.observation_generation(self.x_t))
            return self.observations[:, 0]

        while t > self.object_time:
            self._forward()
            self._set_observation(self.object_time, self.model.observation_generation(self.x_t))

        return self.observations[:, t - self.time_index]

    def save(self, path:str, T:int, quantity:int, prefix:str= 'str', clear_folder=True, bypass_ask = False):
        if self.model.alg != self.model.PF_Type.Bootstrap:
            warn(f'Model is {self.model.alg.name} instead of Bootstrap, are you this is right?')
        if clear_folder:
            if os.path.exists(path):
                
                if bypass_ask:
                    response = 'Y'
                else:
                    print(f'Warning: This will overwrite the directory at path {path}')
                    response = input('Input Y to confirm you want to do this:')
                if response != 'Y' and response != 'y':
                    print('Halting')
                    return
                try:
                    shutil.rmtree(path)
                except:
                    os.remove(path)
            os.mkdir(path)

        for i in range(quantity):
            temp = copy.copy(self)
            Observation_Queue(conversion_object=temp, time_length=T, device=self.device).save(path, i*self.batch_size, prefix, False)


class Observation_Queue(State_Space_Object):
    """State space object act as a queue of observations and (optionally) state vectors
    Reimplements some methods in a simplified way to be more efficient for this
    special case.


    Parameters
    ----------
    xs: (T,s) ndarray or None, default: None
        An array containing the state of dimension s at every time in [0,T].
        If None and ys is not None then observations are not stored.
        Has no effect if ys is None.

    ys: (T, o) ndarray or None, default: None
        An array containing the observations of dimension s at every time in [0,T].
        If None then generate observations from the State_Space_Object
        converstion_object.

    conversion_object: State_Space_Object, default: None
        A state_space_object to have its observations and state (if availiable)
        memorised as a new Observation_Queue object.
        Must not be None if ys is None, using ys to load observations take priority
        otherwise.

    time_length: int or None, default: None
        The number of time steps of conversion_object to memorise
        Has no effect if conversion object is None.
        Must not be None if conversion_object is not None.

    """

    def __init__(self, xs: pt.Tensor = None, ys: pt.Tensor = None, conversion_object: Simulated_Object = None, time_length: int = None, device:str='cuda'):
        self.device = device
        self.object_time = 0
        if ys is not None:
            self.observations = ys
            if xs is not None:
                self.state = xs
            return

        try:
            state_dim = conversion_object.x_t.size()
            self.state = pt.empty((state_dim[0], time_length + 1, state_dim[-1]), device=self.device)
            state_availiable = True
        except AttributeError:
            state_availiable = False

        with pt.inference_mode():

            for t in range(time_length + 1):
                if t == 0:
                    o0 = conversion_object._get_observation(0)
                    self.observations = pt.empty((o0.size(0), time_length + 1, conversion_object.observation_dimension))
                    self.observations[:, 0, :] = o0
                else:
                    self.observations[:,t,:] = conversion_object._get_observation(t)
                
                if state_availiable:
                    self.state[:,t,:] = conversion_object.x_t.squeeze(1)

    def __copy__(self):
        """
        Return a new Observation_Queue with the same
        observations and state set at time 0
        
        """
        try:
            out = Observation_Queue(xs=self.state, ys=self.observations, device=self.device)
        except AttributeError:
            out = Observation_Queue(ys=self.observations, device=self.device)
        return out


    def _get_observation(self, t):
        return self.observations[:, t, :]
    

    def save(self, path:str, start_idx:int, prefix:str='', clear_folder = True) -> None:

        if clear_folder:
            if os.path.exists(path):
                print(f'Warning: This will overwrite the directory at path {path}')
                response = input('Input Y to confirm you want to do this:')
                if response != 'Y' and response != 'y':
                    print('Halting')
                    return
                try:
                    shutil.rmtree(path)
                except:
                    os.remove(path)
            os.mkdir(path)

        for i in range(len(self.observations)):
            pt.save(self.observations[i].clone(), f'{path}/{prefix}_obs_{start_idx+i}_0.pt')
            try:
                pt.save(self.state[i].clone(), f'{path}/{prefix}_state_{start_idx+i}_0.pt')
            except AttributeError:
                pass

    
    
class State_Space_Dataset(pt.utils.data.Dataset):
    """ A custom map style dataset for state space data, appropriate when the data is stored in a single directory.
        Allows for different dimensions of the state or observation data to have different data types, but they must be converted
        to a common type before useage. This is useful when there is e.g. uint8 image data that would be memory intensive to store fully as 
        float32, so it is beneficial to make the conversion at run-time. There is the option to load the data lazily (i.e. only load the data
        to cuda when it is needed) or to pre-load the entire dataset.

    Parameters
    ----------

    path: str
        The path of the directory storing the files

    prefix: str, default: ''
        The prefix of all files

    lazy: bool, default: True
        If true then files are loaded only when required, if false then
        all files are loaded on object creation

    files_per_obs: int, default: 1
        The number of files the observations for each trajectory are stored across,
        assumed to be constant for all trajectories

    files_per_state: int, default: 1
        The number of files the state for each trajectory are stored across,
        assumed to be constant for all trajectories

    obs_data_type: pt.dtype, default: None
        If not None then all observation data will be converted into the given type,
        if left as None then it is left to the pytorch's automatic conversion choice on tensor
        concatenation.

    state_data_type: pt.dtype, default: None
        If not None then all state data will be converted into the given type,
        if left as None then it is left to the pytorch's automatic conversion choice on tensor
        concatenation.

    device: str or pt.device, default: 'cuda'
        The device to put all tensors on.
        
    Notes
    ----------
 
    All files should be 2D pytorch Tensors saved with pt.save()
    The file names may start with an arbitrary but unvarying prefix
    Observations should then be marked 'obs', state marked 'state'
    The files are indexed to link all tensors from the same trajectory
    There is a second index to denote the ordering of tensors to be concatenated for filtering
    e.g. 'directory/prefix_obs_1_1.pt'

    """

    def __init__(self, path:str, prefix:str='', lazy:bool=True, files_per_obs:int=1, files_per_state:int=1, obs_data_type:pt.dtype=None, state_data_type:pt.dtype=None, device:str='cuda', num_workers:int=0) -> None:
        self.lazy = lazy
        self.device = device
        self.length = len([f for f in os.listdir(path) if f.startswith(f'{prefix}_obs')])//files_per_obs
        self.workers = num_workers
    
        if self.lazy:
            self.files_per_obs = files_per_obs
            self.files_per_state = files_per_state
            self.obs_data_type = obs_data_type
            self.state_data_type = state_data_type
            self.prefix = prefix    
            self.dir = path
            return
        
        try:
            self.data = [Observation_Queue(xs = pt.concat(tuple(pt.load(f"{path}/{prefix}_state_{trajectory}_{i}.pt").to(device=device, dtype=state_data_type) for i in range(files_per_state)), dim=-1),
                ys=pt.concat(tuple(pt.load(f"{path}/{prefix}_obs_{trajectory}_{i}.pt").to(device=device, dtype=obs_data_type) for i in range(files_per_obs)), dim=-1)) \
                for trajectory in range(self.length)]
        except FileNotFoundError:
            raise FileNotFoundError('Tensor not found, make sure tensors use the approved naming scheme')
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx:int) -> Observation_Queue:
        if self.lazy:
            try:
                return Observation_Queue(xs = pt.concat(tuple(pt.load(f"{self.dir}/{self.prefix}_state_{idx}_{i}.pt").to(device=self.device, dtype=self.state_data_type) for i in range(self.files_per_state)), dim=-1),
                    ys=pt.concat(tuple(pt.load(f"{self.dir}/{self.prefix}_obs_{idx}_{i}.pt").to(device=self.device, dtype=self.obs_data_type) for i in range(self.files_per_obs)), dim=-1))
            except FileNotFoundError as e:
                print(e)
                raise FileNotFoundError('Tensor not found, make sure tensors use the approved naming scheme')
        return self.data[idx]
    
    def collate(self, batch:Iterable[Observation_Queue]) -> Observation_Queue:
        x_batch = pt.utils.data.default_collate([b.state for b in batch]).to(device=self.device)
        y_batch = pt.utils.data.default_collate([b.observations for b in batch]).to(device=self.device)
        return Observation_Queue(x_batch, y_batch)
    
class dynamic_SS_dataset(pt.utils.data.IterableDataset):
    """Dataset that acts as a wrapper for Simulated_Objects.
    
    Parameters
    -------------
    Template: Simulated_Object
        The simulated object to be duplicated
    
    batch_size: int, default: 1
        The size of the batches data should be generated in
    """

    def __init__(self, template: Simulated_Object, batch_size = 1, num_workers:int=0):
        self.template = copy.copy(template)
        self.template.batches = batch_size
        self.workers = num_workers

    def _generate(self) -> Simulated_Object:
        while True:
            yield copy.copy(self.template)

    def __iter__(self) -> Iterable[Simulated_Object]:
        return iter(self._generate())
    
    def collate(self, batch) -> Simulated_Object:
        if len(batch) != 1:
            warn('Use a dataloader of batch size 1 with the dynamic dataset, the true batch size should be specified at dataset creation')
        return batch[0]