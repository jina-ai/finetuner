# Start fine-tuning

Now you should have your training data and evaluation data (optional) prepared as `DocumentArray`,
and have decided your backbone model.

To start fine-tuning, you can call:

```python
import finetuner
from docarray import DocumentArray

train_data = DocumentArray(...)
model = 'efficientnet_b0'

run = finetuner.fit(
    model=model,
    train_data=train_data
)
```