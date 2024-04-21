import logging

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)


class FeddadaptationsplitServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FeddadaptationsplitServer, self).__init__(**kwargs)
