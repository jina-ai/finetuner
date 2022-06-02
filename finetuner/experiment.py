from dataclasses import fields
from typing import Any, Dict, List, Optional, Union

from docarray import DocumentArray

from finetuner.callbacks import EvaluationCallback
from finetuner.client import FinetunerV1Client
from finetuner.constants import (
    BATCH_SIZE,
    CALLBACKS,
    CONFIG,
    CPU,
    CREATED_AT,
    DATA,
    DESCRIPTION,
    EPOCHS,
    EVAL_DATA,
    EXPERIMENT_NAME,
    FREEZE,
    HYPER_PARAMETERS,
    IMAGE_MODALITY,
    INDEX_DATA,
    LEARNING_RATE,
    LOSS,
    MINER,
    MODEL,
    MULTI_MODAL,
    NAME,
    NUM_WORKERS,
    OPTIMIZER,
    OPTIMIZER_OPTIONS,
    OPTIONS,
    OUTPUT_DIM,
    QUERY_DATA,
    RUN_NAME,
    SCHEDULER_STEP,
    TEXT_MODALITY,
    TRAIN_DATA,
)
from finetuner.hubble import push_data_to_hubble
from finetuner.names import get_random_name
from finetuner.run import Run


class Experiment:
    """Class for an experiment.

    :param client: Client object for sending api requests.
    :param name: Name of the experiment.
    :param status: Status of the experiment.
    :param created_at: Creation time of the experiment.
    :param description: Optional description of the experiment.
    """

    def __init__(
        self,
        client: FinetunerV1Client,
        name: str,
        status: str,
        created_at: str,
        description: Optional[str] = '',
    ):
        self._client = client
        self._name = name
        self._status = status
        self._created_at = created_at
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> str:
        return self._status

    def get_run(self, name: str) -> Run:
        """Get a run by its name.

        :param name: Name of the run.
        :return: A `Run` object.
        """
        run_info = self._client.get_run(experiment_name=self._name, run_name=name)
        run = Run(
            name=run_info[NAME],
            config=run_info[CONFIG],
            created_at=run_info[CREATED_AT],
            description=run_info[DESCRIPTION],
            experiment_name=self._name,
            client=self._client,
        )
        return run

    def list_runs(self) -> List[Run]:
        """List every run inside the experiment.

        :return: List of `Run` objects.
        """
        run_infos = self._client.list_runs(experiment_name=self._name)
        return [
            Run(
                name=run_info[NAME],
                config=run_info[CONFIG],
                created_at=run_info[CREATED_AT],
                description=run_info[DESCRIPTION],
                experiment_name=self._name,
                client=self._client,
            )
            for run_info in run_infos
        ]

    def delete_run(self, name: str):
        """Delete a run by its name.

        :param name: Name of the run.
        """
        self._client.delete_run(experiment_name=self._name, run_name=name)

    def delete_runs(self):
        """Delete every run inside the experiment."""
        self._client.delete_runs(experiment_name=self._name)

    def create_run(
        self,
        model: str,
        train_data: Union[DocumentArray, str],
        run_name: Optional[str] = None,
        **kwargs,
    ) -> Run:
        """Create a run inside the experiment.

        :param model: Name of the model to be fine-tuned.
        :param train_data: Either a `DocumentArray` for training data or a
            name of the `DocumentArray` that is pushed on Hubble.
        :param run_name: Optional name of the run.
        :param kwargs: Optional keyword arguments for the run config.
        :return: A `Run` object.
        """
        if not run_name:
            run_name = get_random_name()

        eval_callback = None
        for callback in kwargs.get(CALLBACKS, []):
            if isinstance(callback, EvaluationCallback):
                eval_callback = callback

        train_data, eval_data = self._prepare_data(
            train_data=train_data,
            eval_data=kwargs.get(EVAL_DATA),
            eval_callback=eval_callback,
            run_name=run_name,
        )
        kwargs[EVAL_DATA] = eval_data

        config = self._create_config_for_run(
            model=model,
            train_data=train_data,
            experiment_name=self._name,
            run_name=run_name,
            **kwargs,
        )

        cpu = kwargs.get(CPU, True)
        num_workers = kwargs.get(NUM_WORKERS, 4)

        run_info = self._client.create_run(
            run_name=run_name,
            experiment_name=self._name,
            run_config=config,
            device='cpu' if cpu else 'gpu',
            cpus=num_workers,
            gpus=1,
        )
        run = Run(
            client=self._client,
            name=run_info[NAME],
            experiment_name=self._name,
            config=run_info[CONFIG],
            created_at=run_info[CREATED_AT],
            description=run_info[DESCRIPTION],
        )
        return run

    def _prepare_data(
        self,
        train_data: Union[DocumentArray, str],
        eval_data: Union[DocumentArray, str],
        eval_callback: EvaluationCallback,
        run_name: str,
    ):
        """Upload data to Hubble and returns their names.

        Uploads all data needed for fine-tuning - training data,
        evaluation data and query/index data for `EvaluationCallback`.

        Checks not to upload same dataset twice.

        :param train_data: Either a `DocumentArray` for training data or a
            name of the `DocumentArray` that is pushed on Hubble.
        :param train_data: Either a `DocumentArray` for evaluation data or a
            name of the `DocumentArray` that is pushed on Hubble.
        :param eval_callback: Evaluation callback that contains query and index data
            and gets modified in-place.
        :param run_name: Name of the run.
        :return: Name(s) of the uploaded data.
        """
        train_data_name = push_data_to_hubble(
            client=self._client,
            data=train_data,
            data_type=TRAIN_DATA,
            experiment_name=self._name,
            run_name=run_name,
        )
        eval_data_name = None
        if eval_data:
            if eval_data == train_data:
                eval_data_name = train_data_name
            else:
                eval_data_name = push_data_to_hubble(
                    client=self._client,
                    data=eval_data,
                    data_type=EVAL_DATA,
                    experiment_name=self._name,
                    run_name=run_name,
                )
        if eval_callback:
            if not eval_callback.query_data:
                eval_callback.query_data = (
                    eval_data_name if eval_data_name else train_data_name
                )
            elif eval_callback.query_data == train_data:
                eval_callback.query_data = train_data_name
            elif eval_callback.query_data == eval_data:
                eval_callback.query_data = eval_data_name
            else:
                eval_callback.query_data = push_data_to_hubble(
                    client=self._client,
                    data=eval_callback.query_data,
                    data_type=QUERY_DATA,
                    experiment_name=self._name,
                    run_name=run_name,
                )

            if not eval_callback.index_data:
                eval_callback.index_data = None
            elif eval_callback.index_data == train_data:
                eval_callback.index_data = train_data_name
            elif eval_callback.index_data == eval_data:
                eval_callback.index_data = eval_data_name
            else:
                eval_callback.index_data = push_data_to_hubble(
                    client=self._client,
                    data=eval_callback.index_data,
                    data_type=INDEX_DATA,
                    experiment_name=self._name,
                    run_name=run_name,
                )
        return train_data_name, eval_data_name

    @staticmethod
    def _create_config_for_run(
        model: str,
        train_data: str,
        experiment_name: str,
        run_name: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create config for a run.

        :param model: Name of the model to be fine-tuned.
        :param train_data: Either a `DocumentArray` for training data or a
            name of the `DocumentArray` that is pushed on Hubble.
        :param experiment_name: Name of the experiment.
        :param run_name: Name of the run.
        :param kwargs: Optional keyword arguments for the run config.
        :return: Run parameters wrapped up as a config dict.
        """
        callbacks = [
            {
                NAME: callback.__class__.__name__,
                OPTIONS: {
                    field.name: getattr(callback, field.name)
                    for field in fields(callback)
                },
            }
            for callback in kwargs.get(CALLBACKS, [])
        ]
        return {
            MODEL: {
                NAME: model,
                FREEZE: kwargs.get(FREEZE),
                OUTPUT_DIM: kwargs.get(OUTPUT_DIM),
                MULTI_MODAL: kwargs.get(MULTI_MODAL),
            },
            DATA: {
                TRAIN_DATA: train_data,
                EVAL_DATA: kwargs.get(EVAL_DATA),
                IMAGE_MODALITY: kwargs.get(IMAGE_MODALITY),
                TEXT_MODALITY: kwargs.get(TEXT_MODALITY),
            },
            HYPER_PARAMETERS: {
                LOSS: kwargs.get(LOSS),
                OPTIMIZER: kwargs.get(OPTIMIZER),
                OPTIMIZER_OPTIONS: {},
                MINER: kwargs.get(MINER),
                BATCH_SIZE: kwargs.get(BATCH_SIZE),
                LEARNING_RATE: kwargs.get(LEARNING_RATE),
                EPOCHS: kwargs.get(EPOCHS),
                SCHEDULER_STEP: kwargs.get(SCHEDULER_STEP),
            },
            CALLBACKS: callbacks,
            EXPERIMENT_NAME: experiment_name,
            RUN_NAME: run_name,
        }
