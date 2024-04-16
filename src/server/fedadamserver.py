import logging
from collections import defaultdict

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)


class FedadamServer(FedavgServer):
    def __init__(self, args, writer, server_dataset, client_datasets, model):
        super(FedavgServer, self).__init__()
        self.args = args
        self.writer = writer

        self.round = 0  # round indicator
        if self.args.eval_type != 'local':  # global holdout set for central evaluation
            self.server_dataset = server_dataset
        self.global_model = self._init_model(model)  # global model
        self.opt_kwargs = dict(
            betas=(self.args.beta1, self.args.beta2),
            v0=self.args.tau ** 2,
            tau=self.args.tau,
            lr=self.args.server_lr
        )
        self.curr_lr = self.args.lr  # learning rate
        self.clients = self._create_clients(client_datasets)  # clients container
        self.results = defaultdict(dict)  # logging results container
        self.server_optimizer = self._get_algorithm(self.global_model, **self.opt_kwargs)
