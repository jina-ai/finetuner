from typing import Any, Dict, List, Optional, Tuple, Union

from docarray import DocumentArray

from finetuner.client.base import BaseClient
from finetuner.constants import (
    API_VERSION,
    ARTIFACT_IDS,
    CONFIG,
    DATA,
    DELETE,
    EVAL_DATA,
    EXPERIMENTS,
    FINETUNED_MODELS_DIR,
    GET,
    MODEL,
    NAME,
    POST,
    RUNS,
    STATUS,
    TRAIN_DATA,
)


class Client(BaseClient):
    def create_experiment(self, name: str, **kwargs) -> dict:
        """Create a new experiment.

        :param name: The name of the experiment.
        :return: Created experiment
        """
        url = self._base_url / API_VERSION / EXPERIMENTS
        return self._handle_request(
            url=url, method=POST, json_data={NAME: name, **kwargs}
        )

    def get_experiment(self, name: str) -> dict:
        """Get an experiment by its name.

        :param name: The name of the experiment.
        :return: Requested experiment
        """
        url = self._base_url / API_VERSION / EXPERIMENTS / name
        return self._handle_request(url=url, method=GET)

    def list_experiments(self) -> List[dict]:
        """List all available experiments.

        :return: List of all experiments
        """
        url = self._base_url / API_VERSION / EXPERIMENTS
        return self._handle_request(url=url, method=GET)

    def delete_experiment(self, name: str) -> dict:
        """Delete an experiment given its name.

        :param name: The name of the experiment.
        :return: Experiment to be deleted
        """
        url = self._base_url / API_VERSION / EXPERIMENTS / name
        return self._handle_request(url=url, method=DELETE)

    def delete_experiments(self) -> List[dict]:
        """Delete all experiments.

        :return: Experiments to be deleted
        """
        url = self._base_url / API_VERSION / EXPERIMENTS
        return self._handle_request(url=url, method=DELETE)

    def create_run(
        self,
        model: str,
        train_data: Union[DocumentArray, str],
        experiment_name: str,
        run_name: str,
        **kwargs,
    ) -> dict:
        """Create a run inside a given experiment.

        For optional parameters please visit our documentation (link).
        :param model: The name of the model to be fine-tuned.
        :param train_data: Either a `DocumentArray` for training data or a
                           name of the `DocumentArray` that is pushed on Hubble.
        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: Created run
        """
        train_data, kwargs[EVAL_DATA] = self._push_data_to_hubble(
            train_data=train_data,
            eval_data=kwargs.get(EVAL_DATA),
            experiment_name=experiment_name,
            run_name=run_name,
        )
        config = self._create_config_for_run(
            model=model, train_data=train_data, **kwargs
        )
        url = self._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
        return self._handle_request(
            url=url, method=POST, json_data={NAME: run_name, CONFIG: config, **kwargs}
        )

    @staticmethod
    def _create_config_for_run(
        model: str,
        train_data: str,
        **kwargs,
    ) -> Dict[str, Any]:
        # Not sure what is the correct way to construct the whole config.
        # Maybe we can create a separate ticket for this.
        config = {}
        config[MODEL] = model
        config[DATA] = {TRAIN_DATA: train_data, EVAL_DATA: kwargs.get(EVAL_DATA)}
        return config

    def get_run(self, experiment_name: str, run_name: str) -> dict:
        """Get a run by its name and experiment.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: Requested run
        """
        url = (
            self._base_url
            / API_VERSION
            / EXPERIMENTS
            / experiment_name
            / RUNS
            / run_name
        )
        return self._handle_request(url=url, method=GET)

    def list_runs(self, experiment_name: Optional[str] = None) -> List[dict]:
        """List all created runs inside a given experiment.

        If no experiment is specified, list runs for all available experiments.
        :param experiment_name: The name of the experiment.
        :return: List of all runs.
        """
        if not experiment_name:
            target_experiments = [
                experiment[NAME] for experiment in self.list_experiments()
            ]
        else:
            target_experiments = [experiment_name]
        response = []
        for experiment_name in target_experiments:
            url = self._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
            response.append(self._handle_request(url=url, method=GET))
        return response

    def delete_run(self, experiment_name: str, run_name: str) -> dict:
        """Delete a run by its name and experiment.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: Deleted run.
        """
        url = (
            self._base_url
            / API_VERSION
            / EXPERIMENTS
            / experiment_name
            / RUNS
            / run_name
        )
        return self._handle_request(url=url, method=DELETE)

    def delete_runs(self, experiment_name: str) -> List[dict]:
        """Delete all runs inside a given experiment.

        :param experiment_name: The name of the experiment.
        :return: List of all deleted runs.
        """
        url = self._base_url / API_VERSION / EXPERIMENTS / experiment_name / RUNS
        return self._handle_request(url=url, method=DELETE)

    def get_run_status(self, experiment_name: str, run_name: str) -> dict:
        """Get a run status by its name and experiment.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: Run status.
        """
        url = (
            self._base_url
            / API_VERSION
            / EXPERIMENTS
            / experiment_name
            / RUNS
            / run_name
            / STATUS
        )
        return self._handle_request(url=url, method=GET)

    def _push_data_to_hubble(
        self,
        train_data: Union[DocumentArray, str],
        eval_data: Optional[Union[DocumentArray, str]],
        experiment_name: str,
        run_name: str,
    ) -> Tuple[str, str]:
        """Push DocumentArray for training and evaluation data on Hubble if it's not already uploaded.

        Note: for now, let's assume that we only receive `DocumentArray`-s.
        :param train_data: Either a `DocumentArray` for training data that needs to be pushed on Hubble
                          or a name of the `DocumentArray` that is already uploaded.
        :param eval_data: Either a `DocumentArray` for evaluation data that needs to be pushed on Hubble
                          or a name of the `DocumentArray` that is already uploaded.
        :returns: Name(s) of pushed `DocumentArray`-s.
        """
        if isinstance(train_data, DocumentArray):
            da_name = '-'.join(
                [self._hubble_user_id, experiment_name, run_name, TRAIN_DATA]
            )
            train_data.push(name=da_name)
            train_data = da_name
        if eval_data and isinstance(eval_data, DocumentArray):
            da_name = '-'.join(
                [self._hubble_user_id, experiment_name, run_name, EVAL_DATA]
            )
            eval_data.push(name=da_name)
            eval_data = da_name
        return train_data, eval_data

    def download_model(
        self, experiment_name: str, run_name: str, path: str = FINETUNED_MODELS_DIR
    ) -> List[str]:
        """Download finetuned model(s) from Hubble by its ID.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :param path: Directory where the model will be stored.
        :returns: A list of str object(s) that indicate the download path on localhost.
        """
        artifact_ids = self.get_run(experiment_name=experiment_name, run_name=run_name)[
            ARTIFACT_IDS
        ]
        response = [
            self._hubble_client.download_artifact(id=artifact_id, path=path)
            for artifact_id in artifact_ids
        ]
        return response
