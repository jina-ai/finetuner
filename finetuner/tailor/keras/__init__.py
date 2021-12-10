from typing import Optional, List, TYPE_CHECKING, Union

import tensorflow as tf

from ..base import BaseTailor

if TYPE_CHECKING:
    from ...helper import LayerInfoType, AnyDNN


class KerasTailor(BaseTailor):
    """Tailor class for Keras DNN models."""

    def summary(self, skip_identity_layer: bool = False) -> 'LayerInfoType':
        """Interpret the DNN model and produce model information.

        :param skip_identity_layer: If skip identity layer.
        :return: The model information stored as dict.
        """

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
        freeze: Union[bool, List[str]] = False,
        bottleneck_net: Optional['AnyDNN'] = None,
    ) -> 'AnyDNN':

        """Convert a general model from :py:attr:`.model` to an embedding model.

        :param layer_name: the name of the layer that is used for output embeddings. All layers *after* that layer
            will be removed. When set to ``None``, then the last layer listed in :py:attr:`.embedding_layers` will be used.
            To see all available names you can check ``name`` field of :py:attr:`.embedding_layers`.
        :param freeze: if set as True, will freeze all layers before :py:`attr`:`layer_name`. If set as list of str, will freeze layers by names.
        :param bottleneck_net: Attach a bottleneck net at the end of model, this module should always trainable.
        :return: Converted embedding model.
        """
        _all_embed_layers = {l['name']: l for l in self.embedding_layers}
        if layer_name:
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

        if _embed_layer != self._model.layers[-1]:
            out = self._model.layers[index].output
            model = tf.keras.Model(self._model.input, out)
        else:
            model = self._model

        if isinstance(freeze, list):
            for layer_name, layer in zip(_all_embed_layers, model.layers):
                if layer_name in freeze:
                    layer.trainable = False
        elif isinstance(freeze, bool) and freeze is True:
            # freeze all layers, not including bottleneck module
            for layer in model.layers:
                layer.trainable = False

        if bottleneck_net:
            # append bottleneck net at the end of embedding model.
            x = model.output
            for layer in bottleneck_net.layers:
                x = layer(x)
            model = tf.keras.Model(model.input, x)

        return model
