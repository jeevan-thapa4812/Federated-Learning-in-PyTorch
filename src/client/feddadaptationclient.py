from .fedavgclient import FedavgClient



class FedDAdaptationClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedDAdaptationClient, self).__init__(**kwargs)