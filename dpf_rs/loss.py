import torch as pt
from torch._tensor import Tensor
from . import results 
from .model import Observation_Queue
from typing import Any, Callable, Iterable
from abc import ABCMeta
import time
from copy import copy

class Loss(metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.create_reporters()

    def register_data(self, **data):
        self.data = data

    def per_step_loss(self, *args) -> pt.Tensor:
        pass

    def per_trajectory_loss(self) -> pt.Tensor:
        return pt.mean(self.per_step_loss(), dim=1)

    def forward(self) -> pt.Tensor:
        return pt.mean(self.per_step_loss())
    
    def backward(self) -> None:
        self.value.backward()

    def item(self) -> float:
        return self.value.item()

    def create_reporters(self) -> Iterable[results.Reporter]:
        self.reporters = []

    def clear_data(self) -> None:
        self.reporters = [copy(r) for r in self.reporters]
    
    def get_reporters(self):
        return self.reporters

    def __call__(self, *args):
        self.value = self.forward(*args)
        return self.value
    
class Compound_Loss(Loss):

    def __init__(self, loss_list:Iterable[Loss]):
        self.loss_list = loss_list
        self.create_reporters()

    def clear_data(self) -> None:
        for loss in self.loss_list:
            loss.clear_data()
        self.create_reporters()

    def create_reporters(self) -> Iterable[results.Reporter]:
        self.reporters = []
        for loss in self.loss_list:
            self.reporters += loss.reporters

    def per_step_loss(self) -> pt.Tensor:
        loss_tensor = pt.stack([loss.per_step_loss() for loss in self.loss_list])
        return pt.einsum('ijk, i', loss_tensor, self.data['weights'])

    def forward(self) -> Tensor:
        loss_tensor = pt.stack([loss() for loss in self.loss_list])
        return pt.dot(loss_tensor, self.data['weights'])       

class Supervised_L2_Loss(Loss):

    def __init__(self, statistic:str='filtering_mean', function:Callable=lambda x:x):
        self.function = function
        self.create_reporters(statistic, function)

    def create_reporters(self, statistic, function):
        if isinstance(statistic, results.Reporter):
            self.reporters = [statistic]
        if statistic == 'filtering_mean':
            self.reporters = [results.Filtering_Mean(function)]
            return
        if statistic == 'predictive_mean':
            self.reporters = [results.True_Predictive_Mean(function)]

    def per_step_loss(self) -> pt.Tensor:
        results = self.function(self.reporters[0].results)
        g_truth = self.function(self.data['truth'].state)[:, :results.size(1), :]
        return pt.sum((results - g_truth)**2, dim=2)

    
class Magnitude_Loss(Loss):
    def __init__(self, statistic, function:Callable=lambda x:x, sign:int=1):
        self.function = function
        self.sign = sign
        self.create_reporters(statistic)
    
    def create_reporters(self, statistic):
        self.reporters = [statistic]

    def per_step_loss(self) -> pt.Tensor:
        results = self.function(self.reporters[0].results)*self.sign
        return pt.sum(results, dim=2)