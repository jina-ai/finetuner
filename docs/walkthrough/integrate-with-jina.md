(integrate-with-jina)=
# Integrate with Jina

Once fine-tuning is finished, it's time to actually use the model. 
Finetuner, being part of the Jina ecosystem, provides a convenient way to use tuned models via [Jina Executors](https://docs.jina.ai/fundamentals/executor/).

We've created the [`FinetunerExecutor`](https://hub.jina.ai/executor/13dzxycc) which can be added in a [Jina Flow](https://docs.jina.ai/fundamentals/flow/) and load any tuned model. 

````{tab} via Docker image (recommended)
```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://FinetunerExecutor', uses_with=...)
```
````
````{tab} via source code
```python
from jina import Flow
	
f = Flow().add(uses='jinahub://FinetunerExecutor', uses_with=...)
```
````

## Using [`FinetunerExecutor`](https://hub.jina.ai/executor/13dzxycc)

Loading a tuned model is simple! You just need to provide a few parameters under the `uses_with` argument when adding the `FinetunerExecutor` to the [Flow]((https://docs.jina.ai/fundamentals/flow/)).

Here's the full argument list of the `FinetunerExecutor`:

```{admonition} FinetunerExecutor parameters
:class: tip
The only required argument is `artifact`. We provide default values for others.
```

* `artifact`: Specify a finetuner run or model artifact. Can be a path to a
        local directory, a path to a local zip file or a Hubble artifact ID. (**required**)
* `token`: A Jina authentication token required for pulling artifacts from
        Hubble. If not provided, the Hubble client will try to find one either in a
        local cache folder or in the environment. (**def=None**)
* `batch_size`: Incoming documents are fed to the graph in batches, both to
        speed-up inference and avoid memory errors. This argument controls the number
        of documents that will be put in each batch. (**def=32**)
* `select_model`: Finetuner run artifacts might contain multiple models. In such
        cases you can select which model to deploy using this argument. (**def=None**)
* `device`: The device to use for inference. Either `cpu` or `cuda`. (**def='cpu'**)
* `device_id`: Specify which CPU or GPU to use for inference. (**def=0**)
* `omp_num_threads`: The number of threads set by OpenMP. Check out
        https://www.openmp.org/spec-html/5.0/openmpse50.html for more information. By
        default it is set to the number of CPUs available. (**def=[cpu_count()](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.cpu_count)**)
* `intra_op_num_threads`: The execution of an individual operation (for some
        operation types) can be parallelized on a pool of `intra_op_num_threads`. 0
        means the system picks an appropriate number. (**def=0**)
* `inter_op_num_threads`: Nodes that perform blocking operations are enqueued
        on a pool of `inter_op_num_threads` available in each process. 0 means
        the system picks an appropriate number. (**def=0**)
* `logging_level`: The executor logging level. See
        https://docs.python.org/3/library/logging.html#logging-levels for available
        options. (**def='DEBUG'**)


## Example
Let's say we want to use a tuned model that is already stored on our local machine inside a [Jina Flow](https://docs.jina.ai/fundamentals/flow/).

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://FinetunerExecutor', 
               uses_with={'artifact': 'model_dir/tuned_model',
                          'batch_size': 16,
                          'device': 'cuda'})
```
 As you can see, it's super easy! We just provided the model path and values for the batch size and gpu inference.