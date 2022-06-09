import json
from typing import Dict, List, Optional, Tuple, Union

from docarray import DocumentArray

from finetuner.constants import DA_PREFIX, FINETUNED_MODEL, MODEL_IDS


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
            train_data, f'{DA_PREFIX}.{experiment_name}.{run_name}.train', _ids2names
        ),
        _push_docarray(
            eval_data, f'{DA_PREFIX}.{experiment_name}.{run_name}.eval', _ids2names
        ),
        _push_docarray(
            query_data, f'{DA_PREFIX}.{experiment_name}.{run_name}.query', _ids2names
        ),
        _push_docarray(
            index_data, f'{DA_PREFIX}.{experiment_name}.{run_name}.index', _ids2names
        ),
    )


def download_model(
    client, experiment_name: str, run_name: str, path: str = FINETUNED_MODEL
) -> List[str]:
    """Download finetuned model(s) from Hubble by its ID.

    :param client: The Finetuner API client.
    :param experiment_name: The name of the experiment.
    :param run_name: The name of the run.
    :param path: Path where the model will be stored.
    :returns: A list of str object(s) that indicate the download path.
    """
    artifact_ids = client.get_run(experiment_name=experiment_name, run_name=run_name)[
        MODEL_IDS
    ]
    artifact_ids = json.loads(artifact_ids)
    if len(artifact_ids) > 1:
        paths = [path + '_' + id for id in artifact_ids]
    else:
        paths = [path]
    response = [
        client.hubble_client.download_artifact(id=artifact_id, path=path)
        for artifact_id, path in zip(artifact_ids, paths)
    ]
    return response
