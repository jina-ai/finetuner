import os
from typing import List, Optional

import hubble
from finetuner.client import FinetunerV1Client
from finetuner.constants import CREATED_AT, DESCRIPTION, NAME, STATUS
from finetuner.experiment import Experiment
from finetuner.run import Run


class Finetuner:
    """Finetuner class."""

    def __init__(self):
        self._client = None
        self._default_experiment = None

    def login(self):
        """Login to Hubble account, initialize a client object
        and create a default experiment.

        Note: Calling `login` is necessary for using finetuner.
        """
        hubble.login()
        self._client = FinetunerV1Client()
        self._default_experiment = self._get_default_experiment()

    @staticmethod
    def _get_cwd() -> str:
        """Returns current working directory."""
        return os.getcwd().split('/')[-1]

    def _get_default_experiment(self) -> Experiment:
        """Create or retrieve (if it already exists) a default experiment
        for the current working directory."""
        experiment_name = self._get_cwd()
        for experiment in self.list_experiments():
            if experiment.name == experiment_name:
                return experiment
        return self.create_experiment(name=experiment_name)

    def create_experiment(self, name: Optional[str] = None) -> Experiment:
        """Create an experiment.

        :param name: Optional name of the experiment. If `None`,
            the experiment is named after the current directory.
        :return: An `Experiment` object.
        """
        if not name:
            name = self._get_cwd()
        experiment_info = self._client.create_experiment(name=name)
        return Experiment(
            client=self._client,
            name=experiment_info[NAME],
            status=experiment_info[STATUS],
            created_at=experiment_info[CREATED_AT],
            description=experiment_info[DESCRIPTION],
        )

    def get_experiment(self, name: str) -> Experiment:
        """Get an experiment by its name.

        :param name: Name of the experiment.
        :return: An `Experiment` object.
        """
        experiment_info = self._client.get_experiment(name=name)
        return Experiment(
            client=self._client,
            name=experiment_info[NAME],
            status=experiment_info[STATUS],
            created_at=experiment_info[CREATED_AT],
            description=experiment_info[DESCRIPTION],
        )

    def list_experiments(self) -> List[Experiment]:
        """List every experiment."""
        experiment_infos = self._client.list_experiments()

        return [
            Experiment(
                client=self._client,
                name=experiment_info[NAME],
                status=experiment_info[STATUS],
                created_at=experiment_info[CREATED_AT],
                description=experiment_info[DESCRIPTION],
            )
            for experiment_info in experiment_infos
        ]

    def delete_experiment(self, name: str) -> Experiment:
        """Delete an experiment by its name.
        :param name: Name of the experiment.
        :return: Deleted experiment.
        """
        experiment_info = self._client.delete_experiment(name=name)
        return Experiment(
            client=self._client,
            name=experiment_info[NAME],
            status=experiment_info[STATUS],
            created_at=experiment_info[CREATED_AT],
            description=experiment_info[DESCRIPTION],
        )

    def delete_experiments(self) -> List[Experiment]:
        """Delete every experiment.
        :return: List of deleted experiments.
        """
        experiment_infos = self._client.delete_experiments()
        return [
            Experiment(
                client=self._client,
                name=experiment_info[NAME],
                status=experiment_info[STATUS],
                created_at=experiment_info[CREATED_AT],
                description=experiment_info[DESCRIPTION],
            )
            for experiment_info in experiment_infos
        ]

    def create_run(
        self,
        model: str,
        train_data,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        **kwargs,
    ) -> Run:
        """Create a run.

        If an experiment name is not specified, the run will be created in the default
        experiment.

        :param model: Name of the model to be fine-tuned.
        :param train_data: Either a `DocumentArray` for training data or a
            name of the `DocumentArray` that is pushed on Hubble.
        :param run_name: Optional name of the run.
        :param experiment_name: Optional name of the experiment.
        :return: A `Run` object.
        """
        if not experiment_name:
            experiment = self._default_experiment
        else:
            experiment = self.get_experiment(name=experiment_name)
        return experiment.create_run(
            model=model, train_data=train_data, run_name=run_name, **kwargs
        )

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

    def list_runs(self, experiment_name: Optional[str] = None) -> List[Run]:
        """List every run.

        If an experiment name is not specified, we'll list every run across all
        experiments.

        :param experiment_name: Optional name of the experiment.
        :return: A list of `Run` objects.
        """
        if not experiment_name:
            experiments = self.list_experiments()
        else:
            experiments = [self.get_experiment(name=experiment_name)]
        runs = []
        for experiment in experiments:
            runs.extend(experiment.list_runs())
        return runs

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
