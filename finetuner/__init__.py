import os
from typing import List, Optional, Union

from docarray import DocumentArray
from dotenv import load_dotenv

from finetuner.constants import (
    DEFAULT_FINETUNER_HOST,
    DEFAULT_HUBBLE_REGISTRY,
    HOST,
    HUBBLE_REGISTRY,
)

if HOST not in os.environ:
    os.environ[HOST] = DEFAULT_FINETUNER_HOST

if HUBBLE_REGISTRY not in os.environ:
    os.environ[HUBBLE_REGISTRY] = DEFAULT_HUBBLE_REGISTRY

from finetuner.experiment import Experiment
from finetuner.finetuner import Finetuner

__version__ = '0.1.0'


load_dotenv()
ft = Finetuner()


def login():
    ft.login()


def list_models():
    """List available models for training."""
    return [
        {
            'resnet50': '''This model was pre-trained on ImageNet dataset, it is recommended to use the model when you want to fine-tune your model on image-to-image search task. The output feature vector dimensionality is 2048.'''
        },
        {
            'resnet152': '''This model was pre-trained on ImageNet dataset, it is recommended to use the model when you want to fine-tune your model on image-to-image search task. The output feature vector dimensionality is 2048.'''
        },
        {
            'efficientnet_b0': '''This model was pre-trained on ImageNet dataset, it is recommended to use the model when you want to fine-tune your model on image-to-image search task. The output feature vector dimensionality is 1280. Compared with ResNet, efficientnet is smaller and faster to train and inference.
            '''
        },
        {
            'efficientnet_b4': '''This model was pre-trained on ImageNet dataset, it is recommended to use the model when you want to fine-tune your model on image-to-image search task. The output feature vector dimensionality is 1792. Compared with ResNet, efficientnet is smaller and faster to train and inference.
            '''
        },
        {
            'openai/clip-vit-base-patch32': '''This model was pre-trained on millions of text image pairs by OpenAI, it is recommended to use the model when you want to fine-tune your model on text-to-image search task. In this case, you directly search image with text with no annotations. The output feature vector dimensionality is 768. It should be noted that fine-tuning CLIP will produce 2 models, a text encoder and an image encoder. Given a text query, you should use text encoder to extract textual features, and pre-compute (offline) visual features using the image encoder.
            '''
        },
        {
            'bert-base-cased': '''This model was pre-trained on BookCorpus and English Wikipedia, it is recommended to use the model when you want to fine-tune your model on text-to-text search task. It is commonly named as dense retrieval. The output feature vector dimensionality is 768.
            '''
        },
        {
            'sentence-transformers/msmarco-distilbert-base-v3': '''This model was pre-trained on BookCorpus and English Wikipedia, it is recommended to use the model when you want to fine-tune your model on text-to-text search task. It is commonly named as dense retrieval. The output feature vector dimensionality is 768. Compared with Bert-base-cased, this model has been fine-tuned on the msmarco dataset.
            '''
        },
    ]


def list_callbacks():
    """List available callbacks."""
    return [
        'BestModelCheckpoint',
        'TrainingCheckpoint',
        'EarlyStopping',
        'WandBLogger',
        'MLFlowLogger',
    ]


def create_experiment(name: Optional[str] = None) -> Experiment:
    """Create an experiment.

    :param name: Optional name of the experiment. If `None`,
        the experiment is named after the current directory.
    :return: An `Experiment` object.
    """
    return ft.create_experiment(name=name)


def get_experiment(name: str) -> Experiment:
    """Get an experiment by its name.

    :param name: Name of the experiment.
    :return: An `Experiment` object.
    """
    return ft.get_experiment(name=name)


def list_experiments() -> List[Experiment]:
    """List every experiment."""
    return ft.list_experiments()


def delete_experiment(name: str):
    """Delete an experiment by its name."""
    ft.delete_experiment(name=name)


def delete_experiments():
    """Delete every experiment."""
    ft.delete_experiments()


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
    callbacks: Optional[list] = None,
    scheduler_step: str = 'batch',
    freeze: bool = False,
    output_dim: Optional[int] = None,
    multi_modal: bool = False,
    image_modality: Optional[str] = None,
    text_modality: Optional[str] = None,
    cpu: bool = True,
    num_workers: int = 4,
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
    :param callbacks: List of callbacks.
    :param scheduler_step: At which interval should the learning rate scheduler's
        step function be called. Valid options are "batch" and "epoch".
    :param freeze: If set to True, will freeze all layers except the last one.
    :param output_dim: The expected output dimension.
        If set, will attach a projection head.
    :param multi_modal: Boolean value, if `True`,
        required argument to the `DataLoader`.
    :param image_modality: The modality of the image `Document`.
    :param text_modality: The modality of the text `Document`.
    :param cpu: Whether to use the CPU. If set to `False` a GPU will be used.
    :param num_workers: Number of CPU workers. If `cpu: False` this is the number of
        workers used by the dataloader.
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
        callbacks=callbacks,
        scheduler_step=scheduler_step,
        freeze=freeze,
        output_dim=output_dim,
        multi_modal=multi_modal,
        image_modality=image_modality,
        text_modality=text_modality,
        cpu=cpu,
        num_workers=num_workers,
    )
    return run
