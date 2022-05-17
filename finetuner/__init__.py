from typing import Optional, Union

from docarray import DocumentArray
from dotenv import load_dotenv

from finetuner.finetuner import Finetuner

load_dotenv()
ft = Finetuner()


def login():
    ft.login()


def list_models():
    """List available models for training."""
    return [
        'resnet50',
        'resnet152',
        'efficientnet_b0',
        'efficientnet_b4',
        'openai/clip-vit-base-patch32',
    ]


def fit(
    model: str,
    train_data: Union[str, DocumentArray],
    eval_data: Optional[Union[str, DocumentArray]] = None,
    run_name: Optional[str] = None,
    description: Optional[str] = None,
    experiment_name: Optional[str] = None,
    loss: str = 'TripletMarginLoss',
    miner: Optional[str] = None,
    optimizer: str = 'Adam',
    learning_rate: float = 0.001,
    epochs: int = 20,
    batch_size: int = 8,
    freeze: bool = False,
    output_dim: Optional[int] = None,
    multi_modal: bool = False,
    image_modality: Optional[str] = None,
    text_modality: Optional[str] = None,
    cpu: bool = False,
    wandb_api_key: Optional[str] = None,
):
    """Start a finetuner run!

    :param model: Name of the model to be fine-tuned.
    :param train_data: Either a `DocumentArray` for training data or a
        name of the `DocumentArray` that is pushed on Hubble.
    :param eval_data: Either a `DocumentArray` for evaluation data or a
        name of the `DocumentArray` that is pushed on Hubble.
    :param run_name: Name of the run.
    :param description: Run description.
    :param experiment_name: Name of the experiment.
    :param loss: Name of the loss function used for fine-tuning.
    :param miner: Name of the miner to create tuple indices for the loss function.
    :param optimizer: Name of the optimizer used for fine-tuning.
    :param learning_rate: learning rate for the optimizer.
    :param epochs: Number of epochs for fine-tuning.
    :param batch_size: Number of items to include in a batch.
    :param freeze: If set to True, will freeze all layers except the last one.
    :param output_dim: The expected output dimension.
        If set, will attach a projection head.
    :param multi_modal: Boolean value, if `True`,
        required argument to the `DataLoader`.
    :param image_modality: The modality of the image `Document`.
    :param text_modality: The modality of the text `Document`.
    :param cpu: Whether to use the CPU, or GPU.
    :param wandb_api_key: WandB key.
    """
    run = ft.create_run(
        model=model,
        train_data=train_data,
        eval_data=eval_data,
        run_name=run_name,
        description=description,
        experiment_name=experiment_name,
        loss=loss,
        miner=miner,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        freeze=freeze,
        output_dim=output_dim,
        multi_modal=multi_modal,
        image_modality=image_modality,
        text_modality=text_modality,
        cpu=cpu,
        wandb_api_key=wandb_api_key,
    )
    return run
