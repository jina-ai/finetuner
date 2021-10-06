from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from jina.helper import cached_property

from ..base import BaseTailor
from ...helper import EmbeddingLayerInfoType


class KerasTailor(BaseTailor):
    @cached_property
    def embedding_layers(self) -> EmbeddingLayerInfoType:
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

    @property
    def output_dim(self) -> int:
        """Get the user-defined output dimensionality.

        :return: Output dimension of the attached linear layer

        .. note::
           if user didn't specify :py:attr:`output_dim`, return model's last layer output dim.
        """
        return self._output_dim or self._model.output_shape[1]

    def _trim(self) -> 'KerasTailor':
        if not self._embedding_layer_name:
            index = self.embedding_layers[-1]['layer_idx']
        else:
            _embed_layers = {l['name']: l for l in self.embedding_layers}
            try:
                index = _embed_layers[self._embedding_layer_name]['layer_idx']
            except KeyError as e:
                raise e
        self._model = Model(self._model.input, self._model.layers[index - 1].output)
        return self

    def _freeze_weights(self) -> 'KerasTailor':
        """Freeze an arbitrary model to make layers not trainable."""
        for layer in self._model.layers:
            layer.trainable = False
        return self

    def _attach_dense_layer(self):
        """Attach a dense layer to the end of the parsed model.

        .. note::
           The attached dense layer have the same shape as the last layer
           in the parsed model.
           The attached dense layer will ignore the :py:attr:`freeze`, this
           layer always trainable.
        """
        if self._output_dim:
            out = Dense(self._output_dim, activation=None, use_bias=True)(
                self._model.layers[-1].output
            )
            self._model = Model(self._model.input, out)
