import json
from typing import List, Union

from docarray import DocumentArray

from finetuner.client import FinetunerV1Client
from finetuner.constants import FINETUNED_MODELS_DIR, MODEL_IDS


def push_data_to_hubble(
    client: FinetunerV1Client,
    data: Union[DocumentArray, str],
    data_type: str,
    experiment_name: str,
    run_name: str,
) -> str:
    """Push DocumentArray for training and evaluation data on Hubble
    if it's not already uploaded.
    Note: for now, let's assume that we only receive `DocumentArray`-s.

    :param client: The Finetuner API client.
    :param data: Either a `DocumentArray` that needs to be pushed on Hubble
        or a name of the `DocumentArray` that is already uploaded.
    :param data_type: Either `TRAIN_DATA` or `EVAL_DATA`.
    :param experiment_name: Name of the experiment.
    :param run_name: Name of the run.
    :returns: Name(s) of pushed `DocumentArray`-s.
    """
    if isinstance(data, DocumentArray):
        da_name = '-'.join(
            [client.hubble_user_id, experiment_name, run_name, data_type]
        )
        data.push(name=da_name)
        data = da_name
    return data


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
    artifact_ids = json.loads(artifact_ids)
    response = [
        client.hubble_client.download_artifact(id=artifact_id, path=path)
        for artifact_id in artifact_ids
    ]
    return response
