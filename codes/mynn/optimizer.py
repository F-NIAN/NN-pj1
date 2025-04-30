from abc import abstractmethod
import numpy as np
from .op import *


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
             if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        grad_with_decay = layer.grads[key] + layer.weight_decay_lambda * layer.params[key] 
                        #l2 reg使用下面
                        #grad_with_decay = layer.layer.grads[key] + layer.weight_decay_lambda * layer.params[key]
                    else:
                        grad_with_decay = layer.grads[key]
                        #l2 reg使用下面
                        #grad_with_decay = layer.layer.grads[key]
                    layer.params[key] -= self.init_lr * grad_with_decay


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocities = [{'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)} 
                          if isinstance(layer, Linear) else None for layer in model.layers]
    
    def step(self):
        for i, layer in enumerate(self.model.layers):
            if not layer.optimizable or self.velocities[i] is None:
                continue
            v = self.velocities[i]
            for key in ['W', 'b']:
                v[key] = self.mu * v[key] + self.init_lr * layer.grads[key]
                layer.params[key] -= v[key]