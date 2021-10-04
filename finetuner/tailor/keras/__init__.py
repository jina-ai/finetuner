from tensorflow.keras import Model

from ..base import BaseTailor
from ...helper import AnyDNN
from ..helper import CandidateLayerInfo


class KerasTailor(BaseTailor):
    def __init__(
        self,
        model: AnyDNN,
        layer_idx: int = -1,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(model, layer_idx, freeze, *args, **kwargs)

    @property
    def candidate_layers(self) -> CandidateLayerInfo:
        """Get all dense layers that can be used as embedding layer from the given model.

        :return: Candidate layers info as list of dictionary.
        """
        results = []
        for idx, layer in enumerate(self._model.layers):
            try:
                output_shape = layer.output_shape
            except AttributeError:
                output_shape = 'multiple'
            except RuntimeError:  # output_shape unknown in Eager mode.
                output_shape = '?'

            if len(output_shape) != 2:
                continue
            else:
                if not layer.built and not getattr(layer, '_is_graph_network', False):
                    # If a subclassed model has a layer that is not called in Model.call, the
                    # layer will not be built and we cannot call layer.count_params().
                    params = 0
                else:
                    params = layer.count_params()

                results.append(
                    {
                        'name': layer.name,
                        'cls_name': layer.__class__.__name__,
                        'output_features': output_shape[-1],
                        'params': params,
                        'layer_idx': idx,
                        'module_name': layer.name,  # duplicate as `name` to make different backends consistent
                    }
                )
        return results

    def _trim(self):
        """Trim an arbitrary Keras model to a Keras embedding model

        ..note::
            The argument `layer_idx` means that all layers before (not include) the index will be
            preserved.
        """
        indx = {l['layer_idx'] for l in self.candidate_layers if l['layer_idx'] != 0}
        if self._layer_idx not in indx:
            raise IndexError(f'Layer index {self._layer_idx} is not one of {indx}.')
        self._model = Model(
            self._model.input, self._model.layers[self._layer_idx - 1].output
        )

    def _freeze_weights(self):
        """Freeze an arbitrary model to make layers not trainable."""
        for layer in self._model.layers:
            layer.trainable = False

    def __call__(self, *args, **kwargs):
        self._trim()
        if self._freze:
            self._freeze_weights()
