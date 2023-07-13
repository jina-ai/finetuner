(pretrained-models)=
# {octicon}`rocket` Jina Embeddings

Starting with Finetuner 0.7.9,
we have introduced a suite of pre-trained text embedding models licensed under Apache 2.0.
The model have a range of useThese models have a variety of use cases, including information retrieval, semantic textual similarity, text reranking, and more.
The suite consists of the following models:

- `jina-embedding-s-en-v1` **[Huggingface](jinaai/jina-embedding-s-en-v1)**: This is a compact model with just 35 million parameters, that performs lightning-fast inference while delivering impressive performance.
- `jina-embedding-b-en-v1` **[Huggingface](jinaai/jina-embedding-b-en-v1)**: This model has a size of 110 million parameters, performs fast inference and delivers better performance than our smaller model.
- `jina-embedding-l-en-v1` **[Huggingface](jinaai/jina-embedding-l-en-v1)**: This is a relatively large model with a size of 330 million parameters, that performs single-gpu inference and delivers better performance than our other model.

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

Jina Embeddings is a suite of language models that have been trained using Jina AI's Linnaeus-Clean dataset.
This dataset consists of 380 million query-document pairs of sentences.
These pairs were obtained from various domains and were carefully selected through a thorough cleaning process.
The Linnaeus-Full dataset, from which the Linnaeus-Clean dataset is derived, originally contained 1.6 billion sentence pairs.

## Characteristics

Each Jina embedding model can encode up to 512 tokens,
with any further tokens being truncated.
The models have different output dimensionalities, as shown in the table below:

|Name|param    |context| Dimension |
|------------------------------|-----|------|-----------|
|jina-embedding-s-en-v1|35m      |512| 512       |
|jina-embedding-b-en-v1|110m      |512| 768       |
|jina-embedding-l-en-v1|330m      |512| 1024      |

## Performance

Please refer to the [Huggingface](jinaai/jina-embedding-s-en-v1) page.
