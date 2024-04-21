import torch

from torch.optim import Optimizer

class DAdaptation(Optimizer):
    def __init__(self, params, **kwargs):
        # lr = kwargs.get('lr')
        # v0 = kwargs.get('v0')
        # tau = kwargs.get('tau')
        # momentum = kwargs.get('betas')
        Optimizer.__init__(self, params=params, defaults={})

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        # TODO: G
        # self.state[param]['G'] = torch.norm(gk, p=2)

        for idx, group in enumerate(self.param_groups):
            gamma_k = 1
            beta = 0.9
            for param in group['params']:
                if param.grad is None:
                    continue
                # get (\Delta_t)
                gk = param.grad.data

                if idx == 0:  # idx == 0: parameters; optimize according to algorithm
                    opt_not_initialized = 'dk' not in self.state[param]
                    if opt_not_initialized:
                        self.state[param]['dk'] = torch.tensor([10e-6])
                        self.state[param]['sk'] = torch.zeros_like(gk).detach()
                        self.state[param]['zk'] = param.data.clone().detach()
                        self.state[param]['lam_g_dot_s_sum'] = 0
                        print('y')

                    lambda_k = self.state[param]['dk'] * gamma_k / self.state[param]['G']

                    # lam_g_dot_s_sum += lambda_k * gk.T @ sk
                    self.state[param]['lam_g_dot_s_sum'] = self.state[param]['lam_g_dot_s_sum'] + lambda_k * torch.dot(
                        gk.view(-1), self.state[param]['sk'].view(-1))

                    # calculate m_t
                    self.state[param]['sk'] = self.state[param]['sk'] + lambda_k * gk  # sk+1 = sk + lambda_k * gk
                    self.state[param]['zk'] = self.state[param]['zk'] - lambda_k * gk  # zk+1 = zk - lambda_k * gk


                    param.data = beta * param.data + (1 - beta) * self.state[param]['zk']

                    dkp1_ = 2 * (self.state[param]['lam_g_dot_s_sum']) / (torch.norm(self.state[param]['sk'], p=2) + 1e-6)
                    self.state[param]['dk'] = torch.max(self.state[param]['dk'], dkp1_)
                elif idx == 1:  # idx == 1: buffers; just averaging
                    param.data.sub_(gk)
        return loss

