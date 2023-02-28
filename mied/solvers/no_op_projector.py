import torch
from mied.solvers.projector_base import ProjectorBase

'''
Handle multiple constraints by projecting the gradients using
Dystra algorithm.
'''
class NoOpProjector(ProjectorBase):
    def __init__(self):
        pass


    def step(self, X, update, problem):
        return update


    def get_violation(self):
        return 0.0
