import os
from typing import Dict, Optional, Tuple, Union

from finetuner import DocumentArray
from finetuner.constants import ARTIFACTS_DIR, DA_PREFIX


def push_docarray(
    data: Union[None, str, DocumentArray],
    name: str,
    ids2names: Optional[Dict[int, str]] = None,
) -> Optional[str]:
    """Upload a DocumentArray to Jina AI Cloud and return its name."""
    if isinstance(data, DocumentArray):
        _id = id(data)  # get the reference id
        if ids2names is not None and _id in ids2names:
            return ids2names[_id]
        print(f'Pushing a DocumentArray to Jina AI Cloud under the name {name} ...')
        data.push(name=name, show_progress=True, public=False)
        if ids2names is not None:
            ids2names[id(data)] = name
        return name
    return data


def push_training_data(
    experiment_name: str,
    run_name: str,
    train_data: Union[str, DocumentArray],
    eval_data: Union[None, str, DocumentArray] = None,
    query_data: Union[None, str, DocumentArray] = None,
    index_data: Union[None, str, DocumentArray] = None,
) -> Tuple[Optional[str], ...]:
    """Upload data to Jina AI Cloud and returns their names.

    Uploads all data needed for fine-tuning - training data,
    evaluation data and query/index data for `EvaluationCallback`.

    Data is given either as a `DocumentArray` or
    a name of the `DocumentArray` that is already pushed to Jina AI Cloud.

    Checks not to upload same dataset twice.

    :param experiment_name: Name of the experiment.
    :param run_name: Name of the run.
    :param train_data: Training data.
    :param eval_data: Evaluation data.
    :param query_data: Query data for `EvaluationCallback`.
    :param index_data: Index data for `EvaluationCallback`.
    :return: Name(s) of the uploaded data.
    """
    _ids2names = dict()
    return (
        push_docarray(
            train_data, f'{DA_PREFIX}-{experiment_name}-{run_name}-train', _ids2names
        ),
        push_docarray(
            eval_data, f'{DA_PREFIX}-{experiment_name}-{run_name}-eval', _ids2names
        ),
        push_docarray(
            query_data, f'{DA_PREFIX}-{experiment_name}-{run_name}-query', _ids2names
        ),
        push_docarray(
            index_data, f'{DA_PREFIX}-{experiment_name}-{run_name}-index', _ids2names
        ),
    )


def push_synthesis_data(
    experiment_name: str,
    run_name: str,
    query_data: Union[str, DocumentArray],
    corpus_data: Union[str, DocumentArray],
) -> Tuple[Optional[str], Optional[str]]:
    """Upload data to Jina AI Cloud and returns their names.

    Uploads all data needed for data synthesis - query data and corpus data.

    Data is given either as a `DocumentArray` or
    a name of the `DocumentArray` that is already pushed to Jina AI Cloud.

    Checks not to upload same dataset twice.

    :param experiment_name: Name of the experiment.
    :param run_name: Name of the run.
    :param query_data: Query data.
    :param corpus_data: Corpus data.
    :return: Names of the uploaded query and corpus data.
    """
    _ids2names = dict()
    return (
        push_docarray(
            query_data, f'{DA_PREFIX}-{experiment_name}-{run_name}-query', _ids2names
        ),
        push_docarray(
            corpus_data, f'{DA_PREFIX}-{experiment_name}-{run_name}-corpus', _ids2names
        ),
    )


def download_artifact(
    client, artifact_id: str, run_name: str, directory: str = ARTIFACTS_DIR
) -> str:
    """Download artifact from Jina AI Cloud by its ID.

    :param client: Hubble client instance.
    :param artifact_id: The artifact id stored in the Jina AI Cloud.
    :param run_name: The name of the run as artifact name to store locally.
    :param directory: Directory where the artifact will be stored.
    :returns: A string that indicates the download path.
    """
    os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, f'{run_name}.zip')

    return client.hubble_client.download_artifact(
        id=artifact_id, f=path, show_progress=True
    )
