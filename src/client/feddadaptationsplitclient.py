from .fedavgclient import FedavgClient



class FeddadaptationsplitClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FeddadaptationsplitClient, self).__init__(**kwargs)