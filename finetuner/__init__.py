from typing import Optional, Union

import hubble
from docarray import DocumentArray
from dotenv import load_dotenv


def login():
    hubble.login()


load_dotenv()


def available_models():
    """List available models for training."""
    return ['resnet50']


def fit(
    model: str,
    train_data: Union[str, DocumentArray],
    eval_data: Optional[Union[str, DocumentArray]] = None,
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
    name: Optional[str] = None,
    description: Optional[str] = None,
    cpu: bool = False,
    wandb_api_key: Optional[str] = None,
):
    """Start a finetuner run!"""
    ...
