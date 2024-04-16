import logging

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)


class FeddadaptationServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FedavgServer, self).__init__(**kwargs)
        self.server_optimizer = self._get_algorithm(self.global_model, **self.opt_kwargs)

    def update(self):
        """Update the global model through federated learning."""
        #################
        # Client Update #
        #################
        # randomly select clients
        selected_ids = self._sample_clients()
        # request update to selected clients
        updated_sizes = self._request(selected_ids, eval=False, participated=True, retain_model=True, save_raw=False)
        # request evaluation to selected clients
        _ = self._request(selected_ids, eval=True, participated=True, retain_model=True, save_raw=False)

        #################
        # Server Update #
        #################

        self.server_optimizer.zero_grad(set_to_none=True)
        self.server_optimizer = self._aggregate(self.server_optimizer, selected_ids,
                                                updated_sizes)  # aggregate local updates
        self.server_optimizer.step()  # update global model with by the aggregated update
        if self.round % self.args.lr_decay_step == 0:  # update learning rate
            self.curr_lr *= self.args.lr_decay
        return selected_ids
