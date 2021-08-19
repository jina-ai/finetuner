# build a simple dense network with bottleneck as 10-dim
import torch.nn as nn

# wrap the user model with our trainer
from trainer.pytorch import PytorchTrainer

# generate artificial positive & negative data
from ..data_generator import fashion_match_doc_generator as fmdg


class UserModel(nn.Module):
    def __init__(self):
        super(UserModel, self).__init__()
        self.act = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
        )

    def forward(self, x):
        output = self.fc(x)
        return output


def test_simple_sequential_model():
    user_model = UserModel()

    pt = PytorchTrainer(user_model, head_layer='CosineLayer')

    # fit and save the checkpoint
    pt.fit(lambda: fmdg(num_total=1000), epochs=5, batch_size=256)
    pt.save('./examples/fashion/trained.pth')
