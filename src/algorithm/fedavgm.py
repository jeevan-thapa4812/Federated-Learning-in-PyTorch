import torch

from .fedavg import FedavgOptimizer


class FedavgmOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(FedavgmOptimizer, self).__init__(params=params, **kwargs)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for idx, group in enumerate(self.param_groups):
            beta = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                delta = param.grad.data

                if idx == 0:  # idx == 0: parameters; optimize according to algorithm // idx == 1: buffers; just averaging
                    if beta > 0.:
                        if 'momentum_buffer' not in self.state[param]:
                            self.state[param]['momentum_buffer'] = torch.zeros_like(param).detach()
                            print('y')

                        # \beta * v + (1 - \beta) * grad
                        self.state[param]['momentum_buffer'].mul_(beta).add_(delta.mul(1. - beta))
                        delta = self.state[param]['momentum_buffer']
                param.data.sub_(delta)
        return loss
