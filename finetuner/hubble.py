from typing import List, Optional, Tuple, Union

from docarray import DocumentArray

from finetuner.client import FinetunerV1Client
from finetuner.constants import EVAL_DATA, FINETUNED_MODELS_DIR, MODEL_IDS, TRAIN_DATA


def push_data_to_hubble(
    client: FinetunerV1Client,
    train_data: Union[DocumentArray, str],
    eval_data: Optional[Union[DocumentArray, str]],
    experiment_name: str,
    run_name: str,
) -> Tuple[str, str]:
    """Push DocumentArray for training and evaluation data on Hubble
    if it's not already uploaded.
    Note: for now, let's assume that we only receive `DocumentArray`-s.

    :param client: The Finetuner API client.
    :param train_data: Either a `DocumentArray` for training data that needs to be
        pushed on Hubble or a name of the `DocumentArray` that is already uploaded.
    :param eval_data: Either a `DocumentArray` for evaluation data that needs to be
        pushed on Hubble or a name of the `DocumentArray` that is already uploaded.
    :param experiment_name: Name of the experiment.
    :param run_name: Name of the run.
    :returns: Name(s) of pushed `DocumentArray`-s.
    """
    if isinstance(train_data, DocumentArray):
        da_name = '-'.join(
            [client.hubble_user_id, experiment_name, run_name, TRAIN_DATA]
        )
        train_data.push(name=da_name)
        train_data = da_name
    if eval_data and isinstance(eval_data, DocumentArray):
        da_name = '-'.join(
            [client.hubble_user_id, experiment_name, run_name, EVAL_DATA]
        )
        eval_data.push(name=da_name)
        eval_data = da_name
    return train_data, eval_data


def download_model(
    client, experiment_name: str, run_name: str, path: str = FINETUNED_MODELS_DIR
) -> List[str]:
    """Download finetuned model(s) from Hubble by its ID.

    :param client: The Finetuner API client.
    :param experiment_name: The name of the experiment.
    :param run_name: The name of the run.
    :param path: Directory where the model will be stored.
    :returns: A list of str object(s) that indicate the download path.
    """
    artifact_ids = client.get_run(experiment_name=experiment_name, run_name=run_name)[
        MODEL_IDS
    ]
    response = [
        client.hubble_client.download_artifact(id=artifact_id, path=path)
        for artifact_id in artifact_ids
    ]
    return response
