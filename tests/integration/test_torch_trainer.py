import os

import torch
import torch.nn as nn
import numpy as np

from trainer.pytorch import PytorchTrainer
from ..data_generator import fashion_match_doc_generator as fmdg

INPUT_DIM = 28
OUTPUT_DIM = 32


class UserModel(nn.Module):
    def __init__(self):
        super(UserModel, self).__init__()
        self.act = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=INPUT_DIM * INPUT_DIM, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=OUTPUT_DIM),
        )

    def forward(self, x):
        return self.fc(x)


def test_simple_sequential_model(tmpdir):
    user_model = UserModel()
    model_path = os.path.join(tmpdir, 'trained.pth')

    pt = PytorchTrainer(user_model, head_layer='CosineLayer')

    # fit and save the checkpoint
    pt.fit(lambda: fmdg(num_total=1000), epochs=2, batch_size=256)
    pt.save(model_path)

    # load the checkpoint and ensure the dim
    embedding_model = torch.load(model_path)
    embedding_model.eval()
    num_samples = 100
    inputs = torch.from_numpy(
        np.random.random([num_samples, INPUT_DIM, INPUT_DIM]).astype(np.float32)
    )
    r = embedding_model(inputs)
    assert r.shape == (num_samples, OUTPUT_DIM)
