# One-liner `fit()`

```{include} ../index.md
:start-after: <!-- start fit-method -->
:end-before: <!-- end fit-method -->
```

## Save model

```python
import finetuner

finetuner.save(model, './saved-model')
```

```{caution}
Depending on your framework, `save` can behave differently. Tensorflow/Keras saves model architecture with the parameters, whereas PyTorch & Paddle only saves the trained parameters.
```

## Display a model

```python
import finetuner

finetuner.display(model)
```

```{caution}
Depending on your framework, `display` may require different argument for rendering the model correctly. In PyTorch & Paddle, you will also need to give the `input_size` and sometimes `input_dtype` to correctly render the model.
```