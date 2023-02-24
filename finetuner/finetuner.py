from typing import Any, Dict, List, Optional, Union

from docarray import DocumentArray

import hubble
from finetuner.client import FinetunerV1Client
from finetuner.constants import CREATED_AT, DESCRIPTION, NAME, STATUS
from finetuner.data import CSVOptions
from finetuner.excepts import FinetunerServerError
from finetuner.experiment import Experiment
from finetuner.run import Run
from hubble import login_required


class Finetuner:
    """Finetuner class."""

    def __init__(self):
        self._client = None
        self._default_experiment = None
        self._default_experiment_name = 'default'

    def login(self, force: bool = False, interactive: Optional[bool] = None):
        """Login to Hubble account, initialize a client object
        and create a default experiment.

        :param force: If set to true, overwrite token and re-login.

        Note: Calling `login` is necessary for using finetuner.
        """
        hubble.login(
            force=force, post_success=self._init_state, interactive=interactive
        )

    @login_required
    def _init_state(self):
        """Initialize client and default experiment."""
        self._client = FinetunerV1Client()
        self._default_experiment = self._get_default_experiment()

    def _get_default_experiment(self) -> Experiment:
        """Create or retrieve (if it already exists) a default experiment
        for the current working directory."""
        for experiment in self.list_experiments():
            if experiment.name == self._default_experiment_name:
                return experiment
        return self.create_experiment(name=self._default_experiment_name)

    @login_required
    def create_experiment(self, name: str = 'default') -> Experiment:
        """Create an experiment.

        :param name: Optional name of the experiment. If `None`,
            the experiment is named after the current directory.
        :return: An `Experiment` object.
        """
        try:
            experiment = self._client.get_experiment(name=name)
        except FinetunerServerError:
            experiment = self._client.create_experiment(name=name)
        return Experiment(
            client=self._client,
            name=experiment[NAME],
            status=experiment[STATUS],
            created_at=experiment[CREATED_AT],
            description=experiment[DESCRIPTION],
        )

    @login_required
    def get_experiment(self, name: str) -> Experiment:
        """Get an experiment by its name.

        :param name: Name of the experiment.
        :return: An `Experiment` object.
        """
        experiment = self._client.get_experiment(name=name)
        return Experiment(
            client=self._client,
            name=experiment[NAME],
            status=experiment[STATUS],
            created_at=experiment[CREATED_AT],
            description=experiment[DESCRIPTION],
        )

    @login_required
    def list_experiments(self, page: int = 1, size: int = 50) -> List[Experiment]:
        """List every experiment.

        :param page: The page index.
        :param size: The number of experiments to retrieve.
        :return: A list of :class:`Experiment` instance.

        ..note:: `page` and `size` works together. For example, page 1 size 50 gives
            the 50 experiments in the first page. To get 50-100, set `page` as 2.
        ..note:: The maximum number for `size` per page is 100.
        """
        experiments = self._client.list_experiments(page=page, size=size)['items']

        return [
            Experiment(
                client=self._client,
                name=experiment[NAME],
                status=experiment[STATUS],
                created_at=experiment[CREATED_AT],
                description=experiment[DESCRIPTION],
            )
            for experiment in experiments
        ]

    @login_required
    def delete_experiment(self, name: str) -> Experiment:
        """Delete an experiment by its name.
        :param name: Name of the experiment.
        :return: Deleted experiment.
        """
        experiment = self._client.delete_experiment(name=name)
        return Experiment(
            client=self._client,
            name=experiment[NAME],
            status=experiment[STATUS],
            created_at=experiment[CREATED_AT],
            description=experiment[DESCRIPTION],
        )

    @login_required
    def delete_experiments(self) -> List[Experiment]:
        """Delete every experiment.
        :return: List of deleted experiments.
        """
        experiments = self._client.delete_experiments()
        return [
            Experiment(
                client=self._client,
                name=experiment[NAME],
                status=experiment[STATUS],
                created_at=experiment[CREATED_AT],
                description=experiment[DESCRIPTION],
            )
            for experiment in experiments
        ]

    @login_required
    def create_run(
        self,
        model: str,
        train_data: Union[str, DocumentArray],
        eval_data: Optional[Union[str, DocumentArray]] = None,
        val_split: float = 0.0,
        model_artifact: Optional[str] = None,
        run_name: Optional[str] = None,
        description: Optional[str] = None,
        experiment_name: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        loss: str = 'TripletMarginLoss',
        miner: Optional[str] = None,
        miner_options: Optional[Dict[str, Any]] = None,
        optimizer: str = 'Adam',
        optimizer_options: Optional[Dict[str, Any]] = None,
        learning_rate: Optional[float] = None,
        epochs: int = 5,
        batch_size: int = 64,
        callbacks: Optional[List[Any]] = None,
        scheduler: Optional[str] = None,
        scheduler_options: Optional[Dict[str, Any]] = None,
        freeze: bool = False,
        output_dim: Optional[int] = None,
        device: str = 'cuda',
        num_workers: int = 4,
        to_onnx: bool = False,
        csv_options: Optional[CSVOptions] = None,
        public: bool = False,
        num_items_per_class: int = 4,
        sampler: str = 'auto',
        loss_optimizer: Optional[str] = None,
        loss_optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> Run:
        """Create a run.

        If an experiment name is not specified, the run will be created in the default
        experiment.

        :return: A `Run` object.
        """
        if not experiment_name:
            experiment = self._default_experiment
        else:
            experiment = self.get_experiment(name=experiment_name)
        return experiment.create_run(
            model=model,
            train_data=train_data,
            eval_data=eval_data,
            val_split=val_split,
            model_artifact=model_artifact,
            run_name=run_name,
            description=description,
            model_options=model_options or {},
            loss=loss,
            miner=miner,
            miner_options=miner_options,
            optimizer=optimizer,
            optimizer_options=optimizer_options,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks or [],
            scheduler=scheduler,
            scheduler_options=scheduler_options,
            freeze=freeze,
            output_dim=output_dim,
            device=device,
            num_workers=num_workers,
            to_onnx=to_onnx,
            csv_options=csv_options,
            public=public,
            num_items_per_class=num_items_per_class,
            sampler=sampler,
            loss_optimizer=loss_optimizer,
            loss_optimizer_options=loss_optimizer_options,
        )

    @login_required
    def get_run(self, run_name: str, experiment_name: Optional[str] = None) -> Run:
        """Get run by its name and (optional) experiment.

        If an experiment name is not specified, we'll look for the run in the default
        experiment.

        :param run_name: Name of the run.
        :param experiment_name: Optional name of the experiment.
        :return: A `Run` object.
        """
        if not experiment_name:
            experiment = self._default_experiment
        else:
            experiment = self.get_experiment(name=experiment_name)
        return experiment.get_run(name=run_name)

    @login_required
    def list_runs(
        self, experiment_name: Optional[str] = None, page: int = 1, size: int = 50
    ) -> List[Run]:
        """List all created runs inside a given experiment.

        If no experiment is specified, list runs for all available experiments.
        :param experiment_name: The name of the experiment.
        :param page: The page index.
        :param size: Number of runs to retrieve.
        :return: List of all runs.

        ..note:: `page` and `size` works together. For example, page 1 size 50 gives
            the 50 runs in the first page. To get 50-100, set `page` as 2.
        ..note:: The maximum number for `size` per page is 100.
        """
        if not experiment_name:
            experiments = self.list_experiments()
        else:
            experiments = [self.get_experiment(name=experiment_name)]
        runs = []
        for experiment in experiments:
            runs.extend(experiment.list_runs(page=page, size=size))
        return runs

    @login_required
    def delete_run(self, run_name: str, experiment_name: Optional[str] = None):
        """Delete a run.

        If an experiment name is not specified, we'll look for the run in the default
        experiment.

        :param run_name: Name of the run.
        :param experiment_name: Optional name of the experiment.
        """
        if not experiment_name:
            experiment = self._default_experiment
        else:
            experiment = self.get_experiment(name=experiment_name)
        experiment.delete_run(name=run_name)

    @login_required
    def delete_runs(self, experiment_name: Optional[str] = None):
        """Delete every run.

        If an experiment name is not specified, we'll delete every run across all
        experiments.

        :param experiment_name: Optional name of the experiment.
        """
        if not experiment_name:
            experiments = self.list_experiments()
        else:
            experiments = [self.get_experiment(name=experiment_name)]
        for experiment in experiments:
            experiment.delete_runs()

    @login_required
    def get_token(self) -> str:
        return hubble.Auth.get_auth_token()
