import math

import torch

from .basealgorithm import BaseOptimizer


class FeddadaptationOptimizer(BaseOptimizer, torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        # lr = kwargs.get('lr')
        # v0 = kwargs.get('v0')
        # tau = kwargs.get('tau')
        # momentum = kwargs.get('betas')
        defaults = dict(gamma_k=0.1, beta=0.9, weight_decay=0, numerator_weighted=0.0, dk=1e-6)
        BaseOptimizer.__init__(self);
        torch.optim.Optimizer.__init__(self, params=params, defaults=defaults)

    def step(self, closure=None):
        group = self.param_groups[0]
        decay = group['weight_decay']
        numerator_weighted = group['numerator_weighted']
        beta = group['beta']
        dk = group['dk']

        gamma_k = max(group['gamma_k'] for group in self.param_groups)

        loss = None
        if closure is not None:
            loss = closure()

        if "G" not in self.state:
            G_sq = 0
            for idx, group in enumerate(self.param_groups):
                for param in group['params']:
                    if param.grad is None:
                        continue
                    gk = param.grad.data

                    if decay != 0:
                        gk.add(param.data, alpha=decay)

                    G_sq += (gk ** 2).sum().item()
            self.state['G'] = math.sqrt(G_sq)

        lambda_k = dk * gamma_k / self.state['G']

        sk_sq = 0

        for idx, group in enumerate(self.param_groups):
            for param in group['params']:
                if param.grad is None:
                    continue

                gk = param.grad.data

                if idx == 0:  # idx == 0: parameters; optimize according to algorithm
                    if 'sk' not in self.state[param]:
                        self.state[param]['sk'] = torch.zeros_like(gk).detach()
                        self.state[param]['zk'] = param.data.clone().detach()

                    if decay != 0:
                        gk.add_(param.data, alpha=decay)

                    self.state[param]['sk'].data.add_(gk, alpha=lambda_k)
                    self.state[param]['zk'].data.sub_(gk, alpha=lambda_k)

                    param.data.mul_(beta).add_(self.state[param]['zk'], alpha=1 - beta)

                    sk_sq += (self.state[param]['sk'] ** 2).sum().item()
                    numerator_weighted += lambda_k * torch.dot(gk.view(-1), self.state[param]['sk'].view(-1))

                elif idx == 1:  # idx == 1: buffers; just averaging
                    param.data.sub_(gk)

        dkp1_hat = (2 * (dk + numerator_weighted) / math.sqrt(sk_sq + 1e-6)).item()
        dk = max(dk, dkp1_hat)
        self.param_groups[0]['dk'] = dk
        self.param_groups[0]['numerator_weighted'] = numerator_weighted
        return loss

    def accumulate(self, mixing_coefficient, local_layers_iterator,
                   check_if=lambda name: 'num_batches_tracked' in name):

        for group in self.param_groups:
            for server_param, (name, local_signals) in zip(group['params'], local_layers_iterator):
                if check_if(name):
                    server_param.data.zero_()
                    server_param.data.grad = torch.zeros_like(server_param)
                    continue
                local_delta = (server_param - local_signals).mul(mixing_coefficient).data.type(server_param.dtype)
                if server_param.grad is None:  # NOTE: grad buffer is used to accumulate local updates!
                    server_param.grad = local_delta
                else:
                    server_param.grad.data.add_(local_delta)
