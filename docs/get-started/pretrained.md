(pretrained-models)=
# {octicon}`rocket` Jina Embeddings

Starting from Finetuner 0.7.9,
we bring a suit of pre-trained text embedding models, includes:

- `jina-embedding-s-en-v1`: With a compact size of just 35 million parameters, the model enables lightning-fast inference while still delivering impressive performance.
- `jina-embedding-b-en-v1`: With a standard size of 110 million parameters, the model enables fast inference while delivering better performance than our small model.
- `jina-embedding-l-en-v1`: With a size of 330 million parameters, the model enables single-gpu inference while delivering better performance than our small and base model.

## Usage

```python
!pip install finetuner
import finetuner

model = finetuner.build_model('jinaai/jina-embedding-l-en-v1')
embeddings = finetuner.encode(
    model=model,
    data=['how is the weather today', 'What is the current weather like today?']
)
print(finetuner.cos_sim(embeddings[0], embeddings[1]))
```

## Training Data

Jina embedding models is a suit of language models that has been trained using Jina AI's Linnaeus-Clean dataset.
This dataset consists of 380 million pairs of sentences, which include both query-document pairs.
These pairs were obtained from various domains and were carefully selected through a thorough cleaning process.
The Linnaeus-Full dataset, from which the Linnaeus-Clean dataset is derived, originally contained 1.6 billion sentence pairs.

## Characteristics

|Name|param    |context| Dimension |
|------------------------------|-----|------|-----------|
|jina-embedding-s-en-v1|35m      |512| 512       |
|jina-embedding-b-en-v1|110m      |512| 768       |
|jina-embedding-l-en-v1|330m      |512| 1024      |

## Performance

The model has a range of use cases, including information retrieval, semantic textual similarity, text reranking, and more.
The table below indicates some key metrics:

|Name|STS12|STS13|STS14|STS15|STS16|STS17|TRECOVID|Quora|SciFact|
|------------------------------|-----|-----|-----|-----|-----|-----|--------|-----|-----|
|jina-embedding-s-en-v1|0.736|0.78|0.745|0.84|0.79|0.868|0.484   |0.856|0.606  |
|jina-embedding-b-en-v1|**0.74**|0.792|0.752|0.851|0.801|0.88|0.505   |0.871|0.64  |
|jina-embedding-l-en-v1|0.736|0.832|0.762|0.846|0.805|0.885|0.477   |**0.876**|0.65  |
