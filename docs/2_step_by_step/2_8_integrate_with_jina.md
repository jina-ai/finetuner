(integrate-with-jina)=
# Integrate with Jina

Once fine-tuning is finished, it's time to actually use the model for your task. 
Finetuner, being part of the Jina ecosystem, provides a convenient way to use tuned models via [Jina Executors](https://docs.jina.ai/fundamentals/executor/).

We've created the [`FinetunerExecutor`](https://hub.jina.ai/executor/13dzxycc) which can be added in a [Jina Flow](https://docs.jina.ai/fundamentals/flow/) and load any tuned model. 

````{tab} via Docker image (recommended)
```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://FinetunerExecutor')
```
````
````{tab} via source code
```python
from jina import Flow
	
f = Flow().add(uses='jinahub://FinetunerExecutor')
```
````

(Since core is private I think this is the only place where developers will be able to see what/how to provide inside this executor, so we should explain each argument and give an example here)
