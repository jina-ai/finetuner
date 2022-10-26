import warnings
from dataclasses import fields
from typing import Any, Dict, List, Optional, Union

from _finetuner.runner.stubs import config
from docarray import DocumentArray

from finetuner.callback import EvaluationCallback
from finetuner.client import FinetunerV1Client
from finetuner.constants import (
    BATCH_SIZE,
    CALLBACKS,
    CONFIG,
    CPU,
    CREATED_AT,
    DESCRIPTION,
    DEVICE,
    EPOCHS,
    EVAL_DATA,
    FREEZE,
    LEARNING_RATE,
    LOSS,
    MINER,
    MINER_OPTIONS,
    MODEL_OPTIONS,
    NAME,
    NUM_WORKERS,
    ONNX,
    OPTIMIZER,
    OPTIMIZER_OPTIONS,
    OUTPUT_DIM,
    SCHEDULER_STEP,
)
from finetuner.hubble import push_data
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
        callbacks = kwargs[CALLBACKS] if kwargs.get(CALLBACKS) else []
        for callback in callbacks:
            if isinstance(callback, EvaluationCallback):
                eval_callback = callback

        train_data, eval_data, query_data, index_data = push_data(
            experiment_name=self._name,
            run_name=run_name,
            train_data=train_data,
            eval_data=kwargs.get(EVAL_DATA),
            query_data=eval_callback.query_data if eval_callback else None,
            index_data=eval_callback.index_data if eval_callback else None,
        )
        if query_data or index_data:
            eval_callback.query_data = query_data
            eval_callback.index_data = index_data

        kwargs[EVAL_DATA] = eval_data

        config = self._create_config_for_run(
            model=model,
            train_data=train_data,
            experiment_name=self._name,
            run_name=run_name,
            **kwargs,
        )

        device = kwargs.get(DEVICE, 'cuda')
        if device == 'cuda':
            device = 'gpu'
            if kwargs.get(CPU, True):
                warnings.warn(
                    message='Parameter `cpu` will be deprecated from Finetuner 0.7.0,'
                    'please use `device="cpu" or `device="cuda" instead.`',
                    category=DeprecationWarning,
                )

        num_workers = kwargs.get(NUM_WORKERS, 4)
        run_info = self._client.create_run(
            run_name=run_name,
            experiment_name=self._name,
            run_config=config,
            device=device,
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
        callbacks = kwargs[CALLBACKS] if kwargs.get(CALLBACKS) else []
        callbacks = [
            config.CallbackConfig(
                name=callback.__class__.__name__,
                options={
                    field.name: getattr(callback, field.name)
                    for field in fields(callback)
                },
            )
            for callback in callbacks
        ]
        model = config.ModelConfig(
            name=model,
            output_dim=kwargs.get(OUTPUT_DIM),
        )
        if kwargs.get(FREEZE):
            model.freeze = kwargs.get(FREEZE)
        if kwargs.get(MODEL_OPTIONS):
            model.options = kwargs.get(MODEL_OPTIONS)
        if kwargs.get(ONNX):
            model.to_onnx = kwargs.get(ONNX)

        data = config.DataConfig(
            train_data=train_data,
            eval_data=kwargs.get(EVAL_DATA),
        )
        if kwargs.get(NUM_WORKERS):
            data.num_workers = kwargs.get(NUM_WORKERS)

        hyper_parameters = config.HyperParametersConfig(
            miner=kwargs.get(MINER),
            learning_rate=kwargs.get(LEARNING_RATE),
        )
        if kwargs.get(LOSS):
            hyper_parameters.loss = kwargs.get(LOSS)
        if kwargs.get(OPTIMIZER):
            hyper_parameters.optimizer = kwargs.get(OPTIMIZER)
        if kwargs.get(OPTIMIZER_OPTIONS):
            hyper_parameters.optimizer_options = kwargs.get(OPTIMIZER_OPTIONS)
        if kwargs.get(MINER_OPTIONS):
            hyper_parameters.miner_options = kwargs.get(MINER_OPTIONS)
        if kwargs.get(BATCH_SIZE):
            hyper_parameters.batch_size = kwargs.get(BATCH_SIZE)
        if kwargs.get(EPOCHS):
            epochs = kwargs.get(EPOCHS)
            hyper_parameters.epochs = epochs
        if kwargs.get(SCHEDULER_STEP):
            hyper_parameters.scheduler_step = kwargs.get(SCHEDULER_STEP)

        run_config = config.RunConfig(
            model=model,
            data=data,
            callbacks=callbacks,
            hyper_parameters=hyper_parameters,
            experiment_name=experiment_name,
            run_name=run_name,
        )

        return run_config.dict()
