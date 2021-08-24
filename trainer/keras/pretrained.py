from typing import List

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.applications as models

from ..pretrained import ModelInterpreter


class KerasModelInterpreter(ModelInterpreter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        weights = 'imagenet' if self._freeze else None
        self.base_model = getattr(models, self._model_name)(weights=weights)
        self._flat_model = None

    @property
    def flat_model(self):
        """Unpack the model architecture recursively and rebuild the model.

        :return: Flattened model.
        """

        def _travese_flat(layers):
            modules = []
            for layer in layers:
                try:
                    modules.extend(_travese_flat(layer.layers))
                except AttributeError:
                    modules.append(layer)

        if not self._flat_model:
            self._flat_model = Sequential(_travese_flat(self.base_model.layers))
        return self._flat_model

    @flat_model.setter
    def flat_model(self, other_model):
        """Unpack the model architecture recursively and rebuild the model.

        :param other_model: Set the current flattened model as other model.
        """
        self._flat_model = other_model

    def _interpret_linear_layers(self):
        """Get all Linear layers inside a model.

        Pytorch use named_modules to get layers recursively.
        :return rv: Dict indicates the layer names and Layer itself,
            to keep the dimensionality of the new layer.
        """
        rv = {}
        for index, module in enumerate(self.flat_model.layers):
            if isinstance(module, Dense):
                rv[index] = module
        return rv

    def _chop_off_last_n_layers(self, layer_index: int):
        """Modify base_model in place based on :attr:`layer_name`.

        Remove last n layers given the layer index, and replace current layer with :class:`nn.Linear`
            with the new :attr:`out_features` as dimensionality.
        :param layer_index: the layer index to remove.
        :return: Modified model.
        """
        pass

    @property
    def trainable_layers(self) -> List[int]:
        """Get trainable layers, e.g. names Linear layers in the backbone model.

        :return: List of linear layer names.
        """
        pass

    def get_modified_base_model(self, layer_index: int):
        """Modify base model based on :attr:`layer_name`. E.g. remove the last n layers
            for retrain.
        :param layer_index: Layer name to modify, if specified, all later layers
            will be removed, the :attr:`out_features` of current layer will be replaced
            with the new output dimension.
        :return: The new base model to be trained.
        """
        pass
