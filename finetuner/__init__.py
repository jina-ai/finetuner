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
        'bert-base-uncased',
        'google/vit-base-patch16-224',
        'openai/clip-vit-base-patch32',
    ]


def fit(
    model: str,
    train_data: Union[str, DocumentArray],
    eval_data: Optional[Union[str, DocumentArray]] = None,
    name: Optional[str] = None,
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
    """Start a finetuner run!"""
    run = ft.create_run(
        model=model,
        train_data=train_data,
        eval_data=eval_data,
        run_name=name,
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
