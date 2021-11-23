# Unlabeled dataset

When using `finetuner.fit(..., interactive=True)`, you only need to provide a `DocumentArray`-like object where each `Document` object contains `.content`. This is because Finetuner will start a web frontend for interactive labeling. Hence, the supervision comes directly from you.

All that the `Document`s in this dataset need is a `.content` attribute (which can be `text`, `blob` or `uri` - if you implement [your own loading](#loading-and-preprocessing)).
