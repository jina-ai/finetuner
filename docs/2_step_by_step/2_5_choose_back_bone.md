# Choose a backbone model

Finetuner prepared several most widely used backbones for you,
including `resnet`, `efficientnet`, `clip` and `bert` etc.

These models has been tested inside Finetuner for tasks such as
image-to-image search, text-to-text search and cross-modal search.

Finetuner will convert these backbone models as embedding models by removing
the *head* or apply *pooling*,
fine-tune and produce the embedding model.

The supported backbones are listed below:

| name                          | task           | output_dim | description                                                                                      |
|-------------------------------|----------------|------------|--------------------------------------------------------------------------------------------------|
| resnet50                      | image-to-image | 2048       | Trained on ImageNet                                                                              |
| resnet152                     | image-to-image | 2048       | Trained on ImageNet                                                                              |
| efficientnet-b0               | image-to-image | 1280       | Trained on ImageNet                                                                              |
| efficientnet-b4               | image-to-image | 1280       | Trained on ImageNet                                                                              |
| openai/clip-vit-base-patch32  | text-to-image  | 768        | Trained on image text pairs from [OpenAI](https://openai.com/blog/clip/)                         |
| bert-base-cased  | text-to-text   | 768        | Trained on BookCorpus and English Wikipedia                                                      |
| sentence-transformers/msmarco-distilbert-base-v3'  | text-to-text   | 768        | Trained based on Bert, fine-tuned on msmarco from [sentence-transformers](https://www.sbert.net/) |