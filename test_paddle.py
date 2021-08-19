import paddle
import paddle.nn.functional as F
import click
paddle.utils.run_check()

# the Document generator
from tests.data_generator import fashion_match_doc_generator as fmdg


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = paddle.nn.Flatten(start_axis=1)
        self.fc1 = paddle.nn.Linear(
            in_features=28 * 28, out_features=128, bias_attr=True
        )
        self.relu = paddle.nn.ReLU()
        self.fc2 = paddle.nn.Linear(in_features=128, out_features=32, bias_attr=True)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

@click.command()
@click.option('--checkpoint_dir', type=str)
def main(checkpoint_dir):
    model = SimpleNet()
    paddle.summary(model, (64, 1, 28, 28))

    from trainer.paddle.trainer import PaddleTrainer

    trainer = PaddleTrainer(base_model=model, head_layer='CosineLayer', checkpoint_dir=checkpoint_dir, use_gpu=True)

    train_data_iter = fmdg(pos_value=1, neg_value=-1)
    trainer.fit(train_data_iter, batch_size=256, shuffle=True, epochs=100)


if __name__ == "__main__":
    main()



