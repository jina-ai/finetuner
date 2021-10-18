import json
from typing import Dict, List


class TunerStats:
    def __init__(
        self,
        loss_train: List = None,
        loss_eval: List = None,
        metrics_train: List[Dict] = None,
        metrics_eval: List[Dict] = None,
    ):
        self._loss_train = loss_train if loss_train is not None else []
        self._loss_eval = loss_eval if loss_eval is not None else []
        self._metrics_train = metrics_train if metrics_train is not None else []
        self._metrics_eval = metrics_eval if metrics_eval is not None else []

    def save(self, file: str):
        with open(file, 'w') as output:
            json.dump(
                {
                    'loss_train': [float(loss) for loss in self._loss_train],
                    'loss_eval': [float(loss) for loss in self._loss_eval],
                    'metrics_train': self._metrics_train,
                    'metrics_eval': self._metrics_eval,
                },
                output,
            )

    def add_train_loss(self, losses: List):
        self._loss_train.extend(losses)

    def add_eval_loss(self, losses: List):
        self._loss_eval.extend(losses)

    def add_train_metric(self, metric: Dict):
        self._metrics_train.append(metric)

    def add_eval_metric(self, metric: Dict):
        self._metrics_eval.append(metric)

    @property
    def last_train_metric(self):
        return self._metrics_train[-1]

    def print_last(self):
        if self._metrics_train and self._metrics_eval:
            train_string = TunerStats.get_metrics_string(self._metrics_train[-1])
            eval_string = TunerStats.get_metrics_string(self._metrics_eval[-1])
            print(f'T: {train_string}, E: {eval_string}')

    @staticmethod
    def get_metrics_string(metrics: Dict):
        return f'hits: {metrics.get("hits", 0):>3}, NDCG: {metrics.get("ndcg", 0):.2f}'
