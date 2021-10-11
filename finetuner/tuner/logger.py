import numpy as np


class LogGenerator:
    def __init__(self, name, losses, metrics, prefix: str = ''):
        self._losses = losses
        self._metrics = metrics
        self._prefix = prefix
        self._name = name

    def __call__(self):
        if self._prefix:
            prefix = f'{self._prefix} | '
        else:
            prefix = ''
        return f'{prefix}{self._name}: {self.get_statistic()}'

    def get_statistic(self):
        return f'L={self.mean_loss():>8} A={self.mean_metric():>4}'

    def mean_loss(self):
        return LogGenerator.get_log_value(self._losses)

    def mean_metric(self):
        return LogGenerator.get_log_value(self._metrics)

    @staticmethod
    def get_log_value(data):
        mean = np.mean(data)
        if mean < 1e5:
            return f'{mean:.2f}'
        else:
            return f'{mean:.2e}'
