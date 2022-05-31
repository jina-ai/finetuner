# Difference between Finetuner and Other Frameworks

There are several fancy machine learning libraries available,
what makes Finetuner unique?

## Care about the quality of Embeddings 🧬

Finetuner is not designed to improve classification,
sentiment analysis or object detection.
Finetuner only care about the quality of embeddings,
and this is what the fine-tuned model will produce.

Given a query `Document` embedded into `embeddings`,
you can compare the similarity/distance of the query `Document`s against all indexed (embedded) `Document`s in your storage backend.


## Dedicated to optimizing your search task 🎯

Finetuner helps you boost your search system performance on different modalities of data:

+ text-to-text search (or dense vector search)
+ image-to-image search (or content-based image search)
+ text-to-image search (based on [OpenAI CLIP](https://openai.com/blog/clip/))
+ more is coming!

Search performance depends on a lot of factors.
Internally we have conducted a lot of experiments on various tasks,
such as image-to-image search,
text-to-text search,
cross-modal search.
Across these three tasks,
**Finetuner is able to boost 20%-45% of precision@k and recall@k**.
You can also observe significant performance improvement on other search metrics,
such as mean recipal rank (mRR) or normalized discounted cumulative gain (nDCG).

## Easy to use 🚀

Finetuner gives user the flexibility to choose machine learning hyper-parameters,
while all these parameters are `Optional`.

If you do not have a machine learning background,
not a problem.
As was stated before, you only need to send us the training data organized as a [DocumentArray](https://docarray.jina.ai/).
In case you do not know which backbone to choose,
use `finetuner.list_models()` to let Finetuner suggest a backbone model for you.