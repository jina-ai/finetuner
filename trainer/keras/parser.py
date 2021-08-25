def get_candidate_layers(model):
    results = []
    for idx, layer in enumerate(model.layers):
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
                }
            )
    return results
