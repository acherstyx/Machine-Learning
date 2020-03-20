from datetime import datetime


class TrainerTemplate:
    def __init__(self, model, data, config):
        """
        init the trainer
        :param model:
        :param data: data loader
        :param config: config you want to use
        """
        self.model = model
        self.data = data
        self.config = config

        self.callbacks = []
        self.metrics = []

        self.timestamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    def train(self):
        """
        train your model here
        """
        raise NotImplementedError
