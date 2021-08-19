import paddle
from paddle import nn
import paddle.nn.functional as F
import click
paddle.utils.run_check()

# the Document generator


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
    user_model = nn.Sequential(
        nn.Flatten(start_axis=1),
        nn.Linear(in_features=784, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=32)
    )
    # user_model = SimpleNet()
    paddle.summary(user_model, (64, 1, 28, 28))

    from trainer.paddle.trainer import PaddleTrainer

    trainer = PaddleTrainer(base_model=user_model, head_layer='CosineLayer', use_gpu=True)

    from tests.data_generator import fashion_match_documentarray as fmdg
    train_data_iter = fmdg(num_total=50)
    trainer.fit(train_data_iter, batch_size=256, shuffle=True, epochs=5)

    from paddle.static import InputSpec
    x_spec = InputSpec(shape=[None, 28, 28], name='x')
    trainer.save('examples/fashion/paddle_ckpt', input_spec=[x_spec])

if __name__ == "__main__":
    main()



