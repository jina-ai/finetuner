import torch

from torch.utils.data._utils.collate import default_collate
from transformers import AutoModel, AutoTokenizer

from finetuner.toydata import generate_qa_match
from finetuner.tuner.pytorch import PytorchTuner


TRANSFORMER_MODEL = 'sentence-transformers/paraphrase-MiniLM-L6-v2'


class TransformerEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(TRANSFORMER_MODEL)

    def forward(self, inputs):
        out_model = self.model(**inputs)
        cls_token = out_model.last_hidden_state[:, 0, :]
        return cls_token


def test_fit_transformer():
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)

    def collate_fn(inputs):
        batch_tokens = tokenizer(
            [x[0] for x in inputs],
            truncation=True,
            max_length=50,
            padding=True,
            return_tensors='pt',
        )
        batch_labels = default_collate([x[1] for x in inputs])
        return batch_tokens, batch_labels

    docs = generate_qa_match(num_neg=8, num_total=100)
    docs_eval = generate_qa_match(num_neg=8, is_testset=True, num_total=20)
    model = TransformerEmbedder()

    tuner = PytorchTuner(model, loss='SiameseLoss')
    tuner.fit(docs, docs_eval, collate_fn=collate_fn, batch_size=30, epochs=2, device='cuda')
