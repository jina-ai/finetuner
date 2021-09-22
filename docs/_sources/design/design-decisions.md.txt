(design-decision)=
# Design Decisions

```{caution}
This section is not meant to be public. It is a collection of my thoughts on the high-level design & positioning of the Finetuner project in the Jina AI landscape. Please do not share any content below.

Some of the paragraph may be published later after my revision.
```

## Single API with minimum mandatory arguments 

Finetuner exposes only a single API: `finetuner.fit`. At its minimum form, it requires only two arguments: the model and the data. 

`Tuner`, `Labeler`, `Tailor` share this single API.

All other arguments are considered as optional and their default values should be carefully chosen by us (i.e. the developer of Finetuner). 


## Three pillars 