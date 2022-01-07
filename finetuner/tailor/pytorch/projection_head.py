import torch.nn as nn


class ProjectionHead(nn.Module):
    """Projection head used internally for self-supervised training.
    It is (by default) a simple 3-layer MLP to be attached on top of embedding model only for training purpose.
    After training, it should be cut-out from the embedding model.
    """

    EPSILON = 1e-5

    def __init__(self, in_features: int, output_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.head_layers = nn.ModuleList()
        is_last_layer = False
        for idx in range(num_layers):
            if idx == num_layers - 1:
                is_last_layer = True
            if not is_last_layer:
                self.head_layers.append(
                    nn.Linear(
                        in_features=in_features, out_features=in_features, bias=False
                    )
                )
                self.head_layers.append(
                    nn.BatchNorm1d(num_features=in_features, eps=self.EPSILON)
                )
                self.head_layers.append(nn.ReLU())
            else:
                self.head_layers.append(
                    nn.Linear(
                        in_features=in_features, out_features=output_dim, bias=False
                    )
                )

    def forward(self, x):
        for layer in self.head_layers:
            x = layer(x)
        return x
