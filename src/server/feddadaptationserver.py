import logging

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)


class FeddadaptationServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FeddadaptationServer, self).__init__(**kwargs)
