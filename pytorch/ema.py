from __future__ import division
from __future__ import unicode_literals

import torch

# https://github.com/fadel/pytorch_ema
# Partially based on: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """
    def __init__(self, named_parameters, decay, device='cuda', use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = {}
        # self.shadow_names = []
        for name, p in named_parameters:
            if p.requires_grad:
                # self.shadow_params.append(p.clone().detach().to(device))
                # self.shadow_names.append(name)
                self.shadow_params[name] = p.clone().detach().to(device)
                print(name, p.data.size())

        # self.shadow_params = [p.clone().detach().to(device)
        #                   for p in parameters if p.requires_grad]
        self.collected_params = {}

    def update(self, named_parameters):
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for name, p in named_parameters:
                if p.requires_grad:
                    if name in self.shadow_params:
                        self.shadow_params[name] = decay * self.shadow_params[name] + one_minus_decay * p
                    else:
                        name = name.replace("module.", "")
                        self.shadow_params[name] = decay * self.shadow_params[name] + one_minus_decay * p
            # parameters = [p for _, p in named_parameters if p.requires_grad]
            # for s_param, param in zip(self.shadow_params, parameters):
            #     s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, named_parameters):
        """
        Copy current parameters into given collection of parameters.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        # for s_param, param in zip(self.shadow_params, parameters):
        #     if param.requires_grad:
        #         print(param.data.size(), s_param.data.size())
        #         param.data.copy_(s_param.data)

        # for nom, s_param in zip(self.shadow_names, self.shadow_params):
        #     named_parameters[nom].data.copy_(s_param.data)
        for name, p in named_parameters:
            if p.requires_grad:
                if name in self.shadow_params:
                    p.data.copy_(self.shadow_params[name].data)
                else:
                    name = name.replace("module.", "")
                    p.data.copy_(self.shadow_params[name].data)

    def store(self, named_parameters):
        """
        Save the current parameters for restoring later.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        # self.collected_params = [param.clone()
        #                          for _, param in named_parameters
        #                          if param.requires_grad]

        for name, param in named_parameters:
            if param.requires_grad:
                self.collected_params[name] = param


    def restore(self, named_parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        # for c_param, param in zip(self.collected_params, named_parameters):
        #     if param[1].requires_grad:
        #         param[1].data.copy_(c_param.data)

        for name, p in named_parameters:
            if p.requires_grad:
                p.data.copy_(self.collected_params[name].data)

        # for name, param in self.collected_params.items():
        #     if param.requires_grad:
        #         named_parameters[name].data.copy_(param.data)
