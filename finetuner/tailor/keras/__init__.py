import copy
from typing import Optional

from jina.helper import cached_property
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from ..base import BaseTailor
from ...helper import EmbeddingLayerInfoType, AnyDNN


class KerasTailor(BaseTailor):
    @cached_property
    def embedding_layers(self) -> EmbeddingLayerInfoType:
        """Get all dense layers that can be used as embedding layer from the :py:attr:`.model`.

        :return: layers info as :class:`list` of :class:`dict`.
        """

        def _get_shape(layer):
            try:
                return layer.output_shape
            except:
                pass  #: return none when

        results = []
        for idx, layer in enumerate(self._model.layers):
            output_shape = _get_shape(layer)
            if (
                not output_shape
                or len(output_shape) != 2
                or not isinstance(output_shape[-1], int)
            ):
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
                        'output_shape': output_shape,
                        'output_features': output_shape[-1],
                        'nb_params': params,
                        'layer_idx': idx,
                        'module_name': layer.name,  # duplicate as `name` to make different backends consistent
                    }
                )
        return results

    def convert(
        self,
        embedding_layer_name: Optional[str] = None,
        output_dim: Optional[int] = None,
        freeze: bool = False,
    ) -> AnyDNN:

        if embedding_layer_name:
            _all_embed_layers = {l['name']: l for l in self.embedding_layers}
            try:
                _embed_layer = _all_embed_layers[embedding_layer_name]
            except KeyError as e:
                raise KeyError(
                    f'`embedding_layer_name` must be one of {_all_embed_layers.keys()}, given {embedding_layer_name}'
                ) from e
        else:
            # when not given, using the last layer
            _embed_layer = self.embedding_layers[-1]

        index = _embed_layer['layer_idx']

        if output_dim:
            out = Dense(output_dim)(self._model.layers[index].output)
        else:
            out = self._model.layers[index].output

        model = Model(self._model.input, out)

        if freeze:
            for layer in model.layers:
                layer.trainable = False

        # the last layer must be trainable
        model.layers[-1].trainable = True
        return model
