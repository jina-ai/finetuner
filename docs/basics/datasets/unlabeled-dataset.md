# Unlabeled Dataset

An unlabeled dataset is a plain `DocumentArray`.

You can use {term}`labeler` to build a {ref}`session-dataset` from it. You only need to provide a `DocumentArray`-like object where each `Document` object contains `.content`, and then call `finetuner.fit(..., interactive=True)`. Finetuner will start a web frontend for interactive labeling.
