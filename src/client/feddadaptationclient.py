from .fedavgclient import FedavgClient



class FeddadaptationClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FeddadaptationClient, self).__init__(**kwargs)