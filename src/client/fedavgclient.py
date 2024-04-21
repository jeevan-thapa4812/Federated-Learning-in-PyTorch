import copy
import inspect
import itertools

import torch

from src import MetricManager
from src.algorithm.dadaptation import DAdaptation
from src.algorithm.dadaptationsplit import DAdaptationSplit
from src.algorithm.dadapt_sgd import DAdaptSGD
from .baseclient import BaseClient


def get_optimizer_class(name):
    if name == "DAdaptation":
        return DAdaptation
    if name == "DAdaptationSplit":
        return DAdaptationSplit
    if name == "DAdaptSGD":
        return DAdaptSGD
    else:
        return torch.optim.__dict__[name]


class FedavgClient(BaseClient):
    def __init__(self, args, training_set, test_set):
        super(FedavgClient, self).__init__()
        self.args = args
        self.training_set = training_set
        self.test_set = test_set

        self.optim = get_optimizer_class(self.args.optimizer)

        # import pdb
        # pdb.set_trace()
        self.use_single_optimizer_across_communication_rounds = True
        # if self.use_single_optimizer_across_communication_rounds:
        #     self.optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))

        self.criterion = torch.nn.__dict__[self.args.criterion]

        self.train_loader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)

    def _refine_optim_args(self, args):
        required_args = inspect.getfullargspec(self.optim)[0]

        # collect eneterd arguments
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument):
                refined_args[argument] = getattr(args, argument)
        return refined_args

    def _create_dataloader(self, dataset, shuffle):
        if self.args.B == 0:
            self.args.B = len(self.training_set)
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=shuffle)

    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)

        # if self.use_single_optimizer_across_communication_rounds:
        #     optimizer = self.optimizer
        # else:
        #     optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))
        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))

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
                optimizer.step()

                mm.track(loss.item(), outputs, targets)
            else:
                mm.aggregate(len(self.training_set), e + 1)
        else:
            self.model.to('cpu')
        return mm.results

    @torch.inference_mode()
    def evaluate(self):
        if self.args.train_only:  # `args.test_size` == 0
            return {'loss': -1, 'metrics': {'none': -1}}

        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)

            mm.track(loss.item(), outputs, targets)
        else:
            self.model.to('cpu')
            mm.aggregate(len(self.test_set))
        return mm.results

    def download(self, model):
        # if self.use_single_optimizer_across_communication_rounds and (self.model is not None):
        #     pass
        self.model = copy.deepcopy(model)

    def upload(self):
        return itertools.chain.from_iterable([self.model.named_parameters(), self.model.named_buffers()])

    def __len__(self):
        return len(self.training_set)

    def __repr__(self):
        return f'CLIENT < {self.id} >'
