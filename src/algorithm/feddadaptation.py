import torch

from .basealgorithm import BaseOptimizer


class FeddadaptationOptimizer(BaseOptimizer, torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        # lr = kwargs.get('lr')
        # v0 = kwargs.get('v0')
        # tau = kwargs.get('tau')
        # momentum = kwargs.get('betas')
        # defaults = dict(lr=lr, momentum=momentum, v0=v0, tau=tau)
        defaults = {}
        BaseOptimizer.__init__(self);
        torch.optim.Optimizer.__init__(self, params=params, defaults=defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for idx, group in enumerate(self.param_groups):
            gamma_k = 1
            beta = 0.9

            for param in group['params']:
                if param.grad is None:
                    continue
                # get (\Delta_t)
                gk = param.grad.data

                if idx == 0:  # idx == 0: parameters; optimize according to algorithm
                    if 'dk' not in self.state[param]:
                        self.state[param]['dk'] = torch.tensor([10e-6])
                        self.state[param]['sk'] = torch.zeros_like(gk)
                        self.state[param]['zk'] = param.data
                        self.state[param]['G'] = torch.norm(gk, p=2)
                        self.state[param]['lam_g_dot_s_sum'] = 0

                    lambda_k = self.state[param]['dk'] * gamma_k / self.state[param]['G']
                    # lam_g_dot_s_sum += lambda_k * gk.T @ sk
                    self.state[param]['lam_g_dot_s_sum'] = self.state[param]['lam_g_dot_s_sum'] + lambda_k * torch.dot(gk.view(-1), self.state[param]['sk'].view(-1))

                    # calculate m_t
                    self.state[param]['sk'] = self.state[param]['sk'] + lambda_k * gk  # sk+1 = sk + lambda_k * gk
                    self.state[param]['zk'] = self.state[param]['zk'] - lambda_k * gk  # zk+1 = zk - lambda_k * gk

                    param.data = beta * param.data + (1 - beta) * self.state[param]['zk']

                    dkp1_ = (self.state[param]['dk'] + self.state[param]['lam_g_dot_s_sum']) / (torch.norm(self.state[param]['sk'], p=2) + 10e-6)
                    self.state[param]['dk'] = torch.max(self.state[param]['dk'], dkp1_)
                elif idx == 1:  # idx == 1: buffers; just averaging
                    param.data.sub_(gk)
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
