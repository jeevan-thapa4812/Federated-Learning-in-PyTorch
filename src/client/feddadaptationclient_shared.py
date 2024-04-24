import copy
import itertools

import torch

from src import MetricManager
from src.algorithm.dadapt_sgd import DAdaptSGD
from src.algorithm.dadaptation import DAdaptation
from src.algorithm.dadaptationsplit import DAdaptationSplit
from .fedavgclient import FedavgClient


def get_optimizer_class(name):
    if name == "DAdaptation":
        return DAdaptation
    if name == "DAdaptationSplit":
        return DAdaptationSplit
    if name == "DAdaptSGD":
        return DAdaptSGD
    else:
        return torch.optim.__dict__[name]


class FeddadaptationClientShared(FedavgClient):
    def __init__(self, **kwargs):
        super(FeddadaptationClientShared, self).__init__(**kwargs)

    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)

        # if self.use_single_optimizer_across_communication_rounds:
        #     optimizer = self.optimizer
        # else:
        #     optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))


        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                outputs = self.model(inputs)
                loss = self.criterion()(outputs, targets)

                for param in self.model.parameters():
                    param.grad = None
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

                mm.track(loss.item(), outputs, targets)
            else:
                mm.aggregate(len(self.training_set), e + 1)
        else:
            self.model.to('cpu')
        return mm.results

    def download(self, model, optimizer_state=None):
        self.model = copy.deepcopy(model)

        if optimizer_state is not None:
            named_optimizer_state = optimizer_state['named_optimizer_state']
            optimizer_init_params = optimizer_state['optimizer_init_params']

            self.optimizer = self.optim(self.model.parameters(),
                                        lr=self.args.lr,
                                        momentum=self.args.momentum,
                                        d0=optimizer_init_params['d'],
                                        k=optimizer_init_params['k'],
                                        numerator_weighted=optimizer_init_params['numerator_weighted']
                                        )

            for name, param in self.model.named_parameters():
                self.optimizer.state[param] = named_optimizer_state[name]

            group = self.optimizer.param_groups[0]
            # group['numerator_weighted'] = optimizer_init_params['numerator_weighted']
            # group['d'] = optimizer_init_params['d']
            group['g0_norm'] = optimizer_init_params['g0_norm']

            group['k'] = optimizer_init_params['k']
            # print("Downloaded g0_norm: ", optimizer_init_params['g0_norm'])
        else:
            self.optimizer = self.optim(self.model.parameters(),
                                        lr=self.args.lr,
                                        momentum=self.args.momentum,)
    def upload(self):
        # extract optimizer state and upload
        group = self.optimizer.param_groups[0]

        named_optimizer_state = {name: self.optimizer.state[param] for name, param in
                                 self.model.named_parameters()}
        optimizer_init_params = {
            'numerator_weighted': group['numerator_weighted'],
            'd': group['d'],
            'g0_norm': group['g0_norm'],
            'k': group['k']
        }

        optimizer_state = {
            'named_optimizer_state': named_optimizer_state,
            'optimizer_init_params': optimizer_init_params
        }
        # print("Uploaded g0_norm: ", optimizer_init_params['g0_norm'])

        del self.optimizer
        return itertools.chain.from_iterable(
            [self.model.named_parameters(), self.model.named_buffers()]), optimizer_state
