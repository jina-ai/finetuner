import abc
from typing import Optional


class ModelParser(abc.ABC):
    def __init__(
        self,
        model_name: str,
        out_features: Optional[int] = 32,
        freeze: bool = True,
        bias: bool = True,
    ):
        self.model_name = model_name
        self.out_features = out_features
        self.freeze = freeze
        self.bias = bias

    @abc.abstractmethod
    def get_modified_base_model(self, layer_index: int):
        ...
