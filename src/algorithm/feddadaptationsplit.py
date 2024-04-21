import torch

from .basealgorithm import BaseOptimizer


class FeddadaptationsplitOptimizer(BaseOptimizer, torch.optim.Optimizer):
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

                gk = param.grad.data

                if idx == 0:  # idx == 0: parameters; optimize according to algorithm
                    if 'dk' not in self.state[param]:
                        self.state[param]['dk'] = torch.tensor([1e-6]).item()
                        self.state[param]['sk'] = torch.zeros_like(gk).detach()
                        self.state[param]['zk'] = param.data.clone().detach()
                        self.state[param]['G'] = torch.norm(gk, p=2).item()
                        self.state[param]['lam_g_dot_s_sum'] = 0
                        print('y')
                    lambda_k = self.state[param]['dk'] * gamma_k / self.state[param]['G']

                    self.state[param]['lam_g_dot_s_sum'] += lambda_k * torch.dot(
                        gk.view(-1), self.state[param]['sk'].view(-1))

                    self.state[param]['sk'].data.add_(gk, alpha=lambda_k)
                    self.state[param]['zk'].data.sub_(gk, alpha=lambda_k)

                    param.data.mul_(beta).add_(self.state[param]['zk'], alpha=1 - beta)

                    dkp1_prev = (self.state[param]['dk'] + self.state[param]['lam_g_dot_s_sum']) / (
                            torch.norm(self.state[param]['sk'], p=2) + 1e-6)

                    dkp1_ = (2 * self.state[param]['lam_g_dot_s_sum'] / (torch.norm(self.state[param]['sk'], p=2) + 1e-6)).item()

                    print("self.state[param]['dk']", "self.state[param]['G']", "lambda_k", "dkp1_prev", "dkp1")
                    print(self.state[param]['dk'], self.state[param]['G'], lambda_k, dkp1_prev, dkp1_)

                    self.state[param]['dk'] = max(self.state[param]['dk'], dkp1_)
                elif idx == 1:  # idx == 1: buffers; just averaging
                    param.data.sub_(gk)
        return loss

    def accumulate(self, mixing_coefficient, local_layers_iterator,
                   check_if=lambda name: 'num_batches_tracked' in name):
        import pdb
        pdb.set_trace()
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
