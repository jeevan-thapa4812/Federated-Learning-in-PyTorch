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


class FeddadaptationClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FeddadaptationClient, self).__init__(**kwargs)

    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)

        # if self.use_single_optimizer_across_communication_rounds:
        #     optimizer = self.optimizer
        # else:
        #     optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))
        self.optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))

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
