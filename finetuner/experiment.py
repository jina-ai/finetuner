from dataclasses import fields
from typing import Any, Dict, List, Optional, TextIO, Union

from _finetuner.runner.stubs import config
from docarray import DocumentArray

from finetuner.callback import EvaluationCallback
from finetuner.client import FinetunerV1Client
from finetuner.constants import (
    BATCH_SIZE,
    CALLBACKS,
    CONFIG,
    CREATED_AT,
    DESCRIPTION,
    DEVICE,
    EPOCHS,
    EVAL_DATA,
    FREEZE,
    LEARNING_RATE,
    LOSS,
    LOSS_OPTIMIZER,
    LOSS_OPTIMIZER_OPTIONS,
    MINER,
    MINER_OPTIONS,
    MODEL_ARTIFACT,
    MODEL_OPTIONS,
    NAME,
    NUM_ITEMS_PER_CLASS,
    NUM_WORKERS,
    ONNX,
    OPTIMIZER,
    OPTIMIZER_OPTIONS,
    OUTPUT_DIM,
    PUBLIC,
    SAMPLER,
    SCHEDULER,
    SCHEDULER_OPTIONS,
    VAL_SPLIT,
)
from finetuner.data import build_finetuning_dataset
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
        """Get the name of the :class:`Experiment`."""
        return self._name

    @property
    def status(self) -> str:
        """Get the status of the :class:`Experiment`."""
        return self._status

    def get_run(self, name: str) -> Run:
        """Get a :class:`Run` given a `name`.

        :param name: Name of the run.
        :return: A `Run` object.
        """
        run = self._client.get_run(experiment_name=self._name, run_name=name)
        run = Run(
            name=run[NAME],
            config=run[CONFIG],
            created_at=run[CREATED_AT],
            description=run[DESCRIPTION],
            experiment_name=self._name,
            client=self._client,
        )
        return run

    def list_runs(self, page: int = 50, size: int = 50) -> List[Run]:
        """List all :class:`Run`.

        :param page: The page index.
        :param size: The number of runs to retrieve per page.
        :return: A list of :class:`Run` instance.

        ..note:: `page` and `size` works together. For example, page 1 size 50 gives
            the 50 runs in the first page. To get 50-100, set `page` as 2.
        ..note:: The maximum number for `size` per page is 100.
        """
        runs = self._client.list_runs(experiment_name=self._name, page=page, size=size)[
            'items'
        ]
        return [
            Run(
                name=run[NAME],
                config=run[CONFIG],
                created_at=run[CREATED_AT],
                description=run[DESCRIPTION],
                experiment_name=self._name,
                client=self._client,
            )
            for run in runs
        ]

    def delete_run(self, name: str):
        """Delete a :class:`Run` by its name.

        :param name: Name of the run.
        """
        self._client.delete_run(experiment_name=self._name, run_name=name)

    def delete_runs(self):
        """Delete all :class:`Run` inside the :class:`Experiment`."""
        self._client.delete_runs(experiment_name=self._name)

    def create_run(
        self,
        model: str,
        train_data: Union[str, TextIO, DocumentArray],
        run_name: Optional[str] = None,
        eval_data: Optional[Union[str, TextIO, DocumentArray]] = None,
        csv_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Run:
        """Create a :class:`Run` inside the :class:`Experiment`."""
        if not run_name:
            run_name = get_random_name()

        eval_callback = None
        callbacks = kwargs[CALLBACKS] if kwargs.get(CALLBACKS) else []
        for callback in callbacks:
            if isinstance(callback, EvaluationCallback):
                eval_callback = callback

        train_data = build_finetuning_dataset(train_data, model, csv_options)

        eval_data = (
            build_finetuning_dataset(eval_data, model, csv_options)
            if eval_data
            else None
        )

        if eval_callback:
            eval_callback.query_data = build_finetuning_dataset(
                eval_callback.query_data, model, csv_options
            )
            eval_callback.index_data = build_finetuning_dataset(
                eval_callback.index_data, model, csv_options
            )

        train_data, eval_data, query_data, index_data = push_data(
            experiment_name=self._name,
            run_name=run_name,
            train_data=train_data,
            eval_data=eval_data,
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

        num_workers = kwargs.get(NUM_WORKERS, 4)
        run = self._client.create_run(
            run_name=run_name,
            experiment_name=self._name,
            run_config=config,
            device=device,
            cpus=num_workers,
            gpus=1,
        )
        run = Run(
            client=self._client,
            name=run[NAME],
            experiment_name=self._name,
            config=run[CONFIG],
            created_at=run[CREATED_AT],
            description=run[DESCRIPTION],
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
        """Create config for a :class:`Run`.

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
        public = kwargs[PUBLIC] if kwargs.get(PUBLIC) else False
        model = config.ModelConfig(
            name=model if not kwargs.get(MODEL_ARTIFACT) else None,
            artifact=kwargs.get(MODEL_ARTIFACT),
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
            val_split=kwargs.get(VAL_SPLIT, 0.0),
            num_items_per_class=kwargs.get(NUM_ITEMS_PER_CLASS, 4),
        )
        if kwargs.get(NUM_WORKERS):
            data.num_workers = kwargs.get(NUM_WORKERS)
        if kwargs.get(SAMPLER):
            data.sampler = kwargs.get(SAMPLER)

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
        if kwargs.get(SCHEDULER):
            hyper_parameters.scheduler = kwargs.get(SCHEDULER)
        if kwargs.get(SCHEDULER_OPTIONS):
            hyper_parameters.scheduler_options = kwargs.get(SCHEDULER_OPTIONS)
        if kwargs.get(LOSS_OPTIMIZER):
            hyper_parameters.loss_optimizer = kwargs.get(LOSS_OPTIMIZER)
        if kwargs.get(LOSS_OPTIMIZER_OPTIONS):
            hyper_parameters.loss_optimizer_options = kwargs.get(LOSS_OPTIMIZER_OPTIONS)

        run_config = config.RunConfig(
            model=model,
            data=data,
            callbacks=callbacks,
            hyper_parameters=hyper_parameters,
            public=public,
            experiment_name=experiment_name,
            run_name=run_name,
        )

        return run_config.dict()
