import abc
from abc import ABC

class BaseModel(ABC):

    @abc.abstractmethod
    def train_step(self):
        """Users can do whatever they need on a single iteration

        """
        ...

    @abc.abstractmethod
    def eval_step(self):
        ...