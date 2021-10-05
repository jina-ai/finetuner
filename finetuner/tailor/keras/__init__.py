from tensorflow.keras import Model

from ..base import BaseTailor
from ...helper import AnyDNN, EmbeddingLayerInfo


class KerasTailor(BaseTailor):
    @property
    def embedding_layers(self) -> EmbeddingLayerInfo:
        """Get all dense layers that can be used as embedding layer from the :py:attr:`.model`.

        :return: layers info as :class:`list` of :class:`dict`.
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
        if not self._embedding_layer_name:
            indx = self.embedding_layers[-1]['layer_idx']
        else:
            _embed_layers = {l['name']: l for l in self.embedding_layers}
            try:
                indx = _embed_layers[self._embedding_layer_name]['layer_idx']
            except KeyError:
                raise KeyError(
                    f'The emebdding layer name {self._embedding_layer_name} does not exist.'
                )

        self._model = Model(self._model.input, self._model.layers[indx].output)

    def _freeze_weights(self):
        """Freeze an arbitrary model to make layers not trainable."""
        for layer in self._model.layers:
            layer.trainable = False

    def __call__(self, *args, **kwargs):
        self._trim()
        if self._freeze:
            self._freeze_weights()
