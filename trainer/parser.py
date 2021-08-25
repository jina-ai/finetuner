import abc
from typing import Optional, Union


class ModelParser(abc.ABC):
    def __init__(
        self,
        base_model: Union[str, 'AnyDnnType'],
        out_features: Optional[int] = None,
        freeze: bool = True,
        bias: bool = True,
    ):
        self.base_model = base_model
        self.out_features = out_features
        self.freeze = freeze
        self.bias = bias

    @abc.abstractmethod
    def get_modified_base_model(self, layer_index: int):
        ...
