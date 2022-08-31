import os
from typing import Dict, Optional, Tuple, Union

from docarray import DocumentArray

from finetuner.constants import ARTIFACTS_DIR, DA_PREFIX


def push_data(
    experiment_name: str,
    run_name: str,
    train_data: Union[str, DocumentArray],
    eval_data: Union[None, str, DocumentArray] = None,
    query_data: Union[None, str, DocumentArray] = None,
    index_data: Union[None, str, DocumentArray] = None,
) -> Tuple[Optional[str], ...]:
    """Upload data to Hubble and returns their names.

    Uploads all data needed for fine-tuning - training data,
    evaluation data and query/index data for `EvaluationCallback`.

    Data is given either as a `DocumentArray` or
    a name of the `DocumentArray` that is already pushed to Hubble.

    Checks not to upload same dataset twice.

    :param experiment_name: Name of the experiment.
    :param run_name: Name of the run.
    :param train_data: Training data.
    :param eval_data: Evaluation data.
    :param query_data: Query data for `EvaluationCallback`.
    :param index_data: Index data for `EvaluationCallback`.
    :return: Name(s) of the uploaded data.
    """

    def _push_docarray(
        data: Union[None, str, DocumentArray], name: str, ids2names: Dict[int, str]
    ) -> Optional[str]:
        if isinstance(data, DocumentArray):
            _id = id(data)  # get the reference id
            if _id in ids2names:
                return ids2names[_id]
            print(f'Pushing a DocumentArray to Hubble under the name {name} ...')
            data.push(name=name, show_progress=True, public=False)
            ids2names[id(data)] = name
            return name
        return data

    _ids2names = dict()
    return (
        _push_docarray(
            train_data, f'{DA_PREFIX}-{experiment_name}-{run_name}-train', _ids2names
        ),
        _push_docarray(
            eval_data, f'{DA_PREFIX}-{experiment_name}-{run_name}-eval', _ids2names
        ),
        _push_docarray(
            query_data, f'{DA_PREFIX}-{experiment_name}-{run_name}-query', _ids2names
        ),
        _push_docarray(
            index_data, f'{DA_PREFIX}-{experiment_name}-{run_name}-index', _ids2names
        ),
    )


def download_artifact(
    client, artifact_id: str, run_name: str, directory: str = ARTIFACTS_DIR
) -> str:
    """Download artifact from Hubble by its ID.

    :param client: Hubble client instance.
    :param artifact_id: The artifact id stored in the Hubble.
    :param run_name: The name of the run as artifact name to store locally.
    :param directory: Directory where the artifact will be stored.
    :returns: A string that indicates the download path.
    """
    os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, f'{run_name}.zip')

    return client.hubble_client.download_artifact(
        id=artifact_id, f=path, show_progress=True
    )
