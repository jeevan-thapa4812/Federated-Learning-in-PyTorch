import math

import torch
import torch.distributed as dist
import torch.optim


class DAdaptSGD(torch.optim.Optimizer):
    r"""
    Implements SGD with D-Adaptation automatic step-sizes. Leave LR set to 1 unless you encounter instability.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the D-adapted learning rate.
        momentum (float):
            Momentum value in  the range [0,1) (default: 0).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate.
            Default is inf, for unrestricted. More conservative values like 1.02 may
            help if training is unstable.
    """

    def __init__(self, params,
                 lr=1.0,
                 momentum=0.0,
                 weight_decay=0,
                 d0=1e-6, growth_rate=float('inf'),
                 ):

        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=1.0,
                        momentum=0.9,
                        weight_decay=weight_decay, k=0,
                        numerator_weighted=0.0,
                        d=d0,
                        growth_rate=growth_rate,
                        )
        self.loggables = {}

        try:
            self.rank = torch.distributed.get_rank()
        except:
            self.rank = 0

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        lr = max(group['lr'] for group in self.param_groups)

        decay = group['weight_decay']
        momentum = group['momentum']
        ck = 1 - momentum
        k = group['k']

        numerator_weighted = group['numerator_weighted']
        growth_rate = group['growth_rate']
        d = group['d']

        group = self.param_groups[0]

        sk_sq = 0.0

        if k == 0:
            g_sq = 0.0
            for group in self.param_groups:
                group_lr = group['lr']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data

                    # Apply weight decay
                    if decay != 0:
                        grad.add(p.data, alpha=decay)

                    state = self.state[p]

                    if group_lr > 0.0:
                        g_sq += (grad * grad).sum().item()

            global_gsq = g_sq
            group['g0_norm'] = math.sqrt(global_gsq)

        g0_norm = group['g0_norm']

        dlr = d * lr / g0_norm

        for group in self.param_groups:
            group_lr = group['lr']
            if group_lr not in [lr, 0.0]:
                raise RuntimeError(
                    f"Setting different lr values in different parameter groups is only supported for values of 0")

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'z' not in state:
                    z = state['z'] = torch.clone(p.data).detach()
                    s = state['s'] = torch.zeros_like(p.data).detach()
                    x0 = state['x0'] = torch.clone(p.data).detach()

                # Apply weight decay
                if decay != 0:
                    grad.add_(p.data, alpha=decay)

                s = state['s']

                if group_lr > 0.0:
                    numerator_weighted += dlr * torch.dot(grad.flatten(), s.flatten()).item()

                    s.data.add_(grad, alpha=dlr)
                    sk_sq += (s * s).sum().item()
            ######


        if lr > 0.0:
            global_sk_sq = sk_sq
            global_numerator_weighted = numerator_weighted

            d_hat = 2 * global_numerator_weighted / math.sqrt(global_sk_sq)
            d = max(d, min(d_hat, d * growth_rate))

        # if we have not done any updates
        # if we have any gradients available, will have sk_sq > 0 (unless \|g\|=0)
        if global_sk_sq == 0:
            return loss

        for group in self.param_groups:
            group['numerator_weighted'] = numerator_weighted
            group['d'] = d
            group['g0_norm'] = g0_norm
            ######################################
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                s = state['s']
                x0 = state['x0']
                z = state['z']

                # z step
                z.data.copy_(x0 - s)

                # x step
                p.data.mul_(1 - ck).add_(z, alpha=ck)

            group['k'] = k + 1

        return loss