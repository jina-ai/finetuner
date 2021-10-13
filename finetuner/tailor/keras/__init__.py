from typing import Optional

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from ..base import BaseTailor
from ...helper import LayerInfoType, AnyDNN


class KerasTailor(BaseTailor):
    """Tailor class for Keras DNN models."""

    def summary(self, skip_identity_layer: bool = False) -> LayerInfoType:
        def _get_output_shape(layer):
            try:
                return layer.output_shape
            except:
                pass  #: return none when

        def _get_input_shape(layer):
            try:
                return layer.input_shape
            except:
                pass  #: return none when

        results = []
        for idx, layer in enumerate(self._model.layers):
            output_shape = _get_output_shape(layer)
            input_shape = _get_input_shape(layer)
            is_embedding_layer = not (
                not output_shape
                or len(output_shape) != 2
                or not isinstance(output_shape[-1], int)
            )

            if not layer.built and not getattr(layer, '_is_graph_network', False):
                # If a subclassed model has a layer that is not called in Model.call, the
                # layer will not be built and we cannot call layer.count_params().
                params = 0
            else:
                params = layer.count_params()

            if skip_identity_layer and output_shape == input_shape and not params:
                # not an effective layer, often a wrapper/identity layer
                continue

            results.append(
                {
                    'name': layer.name,
                    'cls_name': layer.__class__.__name__,
                    'input_shape': input_shape,
                    'output_shape': output_shape,
                    'output_shape_display': list(output_shape[1:]),
                    'output_features': output_shape[
                        -1
                    ],  #: this only makes sense when is_embedding_layer is True
                    'nb_params': params,
                    'layer_idx': idx,
                    'module_name': layer.name,  # duplicate as `name` to make different backends consistent
                    'is_embedding_layer': is_embedding_layer,
                    'trainable': layer.trainable if params else False,
                }
            )
        return results

    def to_embedding_model(
        self,
        layer_name: Optional[str] = None,
        output_dim: Optional[int] = None,
        freeze: bool = False,
    ) -> AnyDNN:

        if layer_name:
            _all_embed_layers = {l['name']: l for l in self.embedding_layers}
            try:
                _embed_layer = _all_embed_layers[layer_name]
            except KeyError as e:
                raise KeyError(
                    f'`embedding_layer_name` must be one of {_all_embed_layers.keys()}, given {layer_name}'
                ) from e
        else:
            # when not given, using the last layer
            _embed_layer = self.embedding_layers[-1]

        index = _embed_layer['layer_idx']

        if output_dim:
            out = Dense(output_dim)(self._model.layers[index].output)
            model = Model(self._model.input, out)
        elif _embed_layer != self._model.layers[-1]:
            out = self._model.layers[index].output
            model = Model(self._model.input, out)
        else:
            model = self._model

        if freeze:
            for layer in model.layers:
                layer.trainable = False

        # the last layer must be trainable
        model.layers[-1].trainable = True
        return model
