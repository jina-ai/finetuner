(pretrained-models)=
# {octicon}`rocket` Jina Embeddings

Starting with Finetuner 0.7.9,
we introduce a suite of pre-trained text embedding models licensed under Apache 2.0.
The model have a range of use cases, including information retrieval, semantic textual similarity, text reranking, and more.
The suite includes the following:

- `jina-embedding-s-en-v1` **[Huggingface](jinaai/jina-embedding-s-en-v1)**: With a compact size of just 35 million parameters, the model enables lightning-fast inference while still delivering impressive performance.
- `jina-embedding-b-en-v1` **[Huggingface](jinaai/jina-embedding-b-en-v1)**: With a standard size of 110 million parameters, the model enables fast inference while delivering better performance than our small model.
- `jina-embedding-l-en-v1` **[Huggingface](jinaai/jina-embedding-l-en-v1)**: With a size of 330 million parameters, the model enables single-gpu inference while delivering better performance than our small and base model.

## Usage

```python
import finetuner

model = finetuner.build_model('jinaai/jina-embedding-s-en-v1')
embeddings = finetuner.encode(
    model=model,
    data=['how is the weather today', 'What is the current weather like today?']
)
print(finetuner.cos_sim(embeddings[0], embeddings[1]))
```

## Training Data

Jina embedding models is a suit of language models that have been trained using Jina AI's Linnaeus-Clean dataset.
This dataset consists of 380 million pairs of sentences, which include both query-document pairs.
These pairs were obtained from various domains and were carefully selected through a thorough cleaning process.
The Linnaeus-Full dataset, from which the Linnaeus-Clean dataset is derived, originally contained 1.6 billion sentence pairs.

## Characteristics

Each Jina embedding model allows encoding of up to 512 tokens,
with any additional tokens being truncated.
The output dimensionality varies across different models.
Please consult the table below for more details.

|Name|param    |context| Dimension |
|------------------------------|-----|------|-----------|
|jina-embedding-s-en-v1|35m      |512| 512       |
|jina-embedding-b-en-v1|110m      |512| 768       |
|jina-embedding-l-en-v1|330m      |512| 1024      |

## Performance

Please refer to the [Huggingface](jinaai/jina-embedding-s-en-v1) page.
