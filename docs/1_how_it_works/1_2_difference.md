# Comparison to other frameworks

There are several fancy machine learning libraries out there,
so what makes Finetuner unique?

## Focus on the quality of embeddings

Finetuner is not designed to tackle classification,
sentiment analysis or object detection task.
Finetuner cares about the quality of the embeddings for neural search,
and this is what the fine-tuned model will produce.

Given a query {class}`~docarray.document.Document` represented by `embeddings`,
you can compare the similarity/distance of the query Documents against all indexed (embedded) Documents in your storage backend.


## Dedicated to optimizing your search task

Finetuner helps you boost your search system performance on different uses cases:

+ text-to-text search (or dense vector search)
+ image-to-image search (or content-based image search)
+ text-to-image search (based on [OpenAI CLIP](https://openai.com/blog/clip/))
+ more is on the way!

Search performance depends on a lot of factors.
Internally we have conducted a lot of experiments on various tasks,
such as image-to-image search,
text-to-text search,
cross-modal search.
Across these three tasks,
**Finetuner is able to boost 20%-45% of precision@k and recall@k**.
You can also observe significant performance improvement on other search metrics,
such as mean recipal rank (mRR) or normalized discounted cumulative gain (nDCG).

## Easy to use

Finetuner gives the user flexibility to choose machine learning hyper-parameters,
while all these parameters are optional.

If you do not have a machine learning background,
don't worry about it.
As was stated before, you only need to provide the training data organized as a {class}`~docarray.array.document.DocumentArray`.
In case you do not know which backbone to choose,
use {meth}`~finetuner.describe_models()` to let Finetuner suggest a backbone model for you.