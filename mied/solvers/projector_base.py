from abc import ABC, abstractmethod
import torch

class ProjectorBase:
    def __init__(self):
        pass


    @abstractmethod
    def step(self, particles, update_grad, problem, optimizer):
        '''
        Update particles given update directions update_grad while projecting
        to the constraints given by the problem.
        '''
        pass


    @abstractmethod
    def get_violation(self):
        pass
