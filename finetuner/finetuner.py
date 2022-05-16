import os
from typing import List, Optional

import hubble
from finetuner.client import FinetunerV1Client
from finetuner.constants import CREATED_AT, DESCRIPTION, NAME, STATUS
from finetuner.experiment import Experiment
from finetuner.names import get_random_name
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

    def _get_default_experiment(self) -> Experiment:
        """Create or retrieve (if it already exists) a default experiment
        for the current working directory."""
        experiment_name = os.getcwd().split('/')[-1]
        for experiment in self.list_experiments():
            if experiment.get_name() == experiment_name:
                return experiment
        experiment = self.create_experiment(name=experiment_name)
        return experiment

    def create_experiment(self, name: Optional[str] = None) -> Experiment:
        """Create an experiment.

        :param: Optional name of the experiment.
        :returns: An `Experiment` object.
        """
        if not name:
            name = get_random_name()
        experiment_info = self._client.create_experiment(name=name)
        experiment = Experiment(
            client=self._client,
            name=experiment_info[NAME],
            status=experiment_info[STATUS],
            created_at=experiment_info[CREATED_AT],
            description=experiment_info[DESCRIPTION],
        )
        return experiment

    def get_experiment(self, name: str) -> Experiment:
        """Get an experiment by its name.

        :param name: Name of the experiment.
        :returns: An `Experiment` object.
        """
        experiment_info = self._client.get_experiment(name=name)
        experiment = Experiment(name=experiment_info[NAME], client=self._client)
        return experiment

    def list_experiments(self) -> List[Experiment]:
        """List every experiment."""
        experiment_infos = self._client.list_experiments()
        experiments = [
            Experiment(
                client=self._client,
                name=experiment_info[NAME],
                status=experiment_info[STATUS],
                created_at=experiment_info[CREATED_AT],
            )
            for experiment_info in experiment_infos
        ]
        return experiments

    def delete_experiment(self, name: str):
        """Delete an experiment by its name."""
        self._client.delete_experiment(name=name)

    def delete_experiments(self):
        """Delete every experiment."""
        self._client.delete_experiments()

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
        :returns: A `Run` object.
        """
        if not experiment_name:
            experiment = self._default_experiment
        else:
            experiment = self.get_experiment(name=experiment_name)
        run = experiment.create_run(
            model=model, train_data=train_data, run_name=run_name, **kwargs
        )
        return run

    def get_run(self, run_name: str, experiment_name: Optional[str] = None) -> Run:
        """Get run by its name and (optional) experiment.

        If an experiment name is not specified, we'll look for the run in the default
        experiment.

        :param run_name: Name of the run.
        :param experiment_name: Optional name of the experiment.
        :returns: A `Run` object.
        """
        if not experiment_name:
            experiment = self._default_experiment
        else:
            experiment = self.get_experiment(name=experiment_name)
        run = experiment.get_run(name=run_name)
        return run

    def list_runs(self, experiment_name: Optional[str] = None) -> List[Run]:
        """List every run.

        If an experiment name is not specified, we'll list every run across all
        experiments.

        :param experiment_name: Optional name of the experiment.
        :returns: A list of `Run` objects.
        """
        if not experiment_name:
            experiments = self.list_experiments()
        else:
            experiments = [experiment_name]
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
            experiments = [experiment_name]
        for experiment in experiments:
            experiment.delete_runs()
