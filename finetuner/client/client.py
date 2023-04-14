from typing import Any, Dict, Iterator, List, Optional

import pkg_resources
from requests.exceptions import ReadTimeout

from finetuner.client.base import _BaseClient
from finetuner.constants import (
    API_VERSION,
    CONFIG,
    CPUS,
    DELETE,
    DESCRIPTION,
    DEVICE,
    EXAMPLES,
    EXPERIMENTS,
    FINETUNER_VERSION,
    GET,
    GPUS,
    LOGS,
    LOGSTREAM,
    METRICS,
    NAME,
    POST,
    RUNS,
    STATUS,
    TASK,
)

_finetuner_core_version = pkg_resources.get_distribution('finetuner-stubs').version


class FinetunerV1Client(_BaseClient):
    """
    The Finetuner v1 API client.
    """

    """ Experiment API """

    def create_experiment(
        self, name: str = 'default', description: Optional[str] = ''
    ) -> Dict[str, Any]:
        """Create a new experiment.

        :param name: The name of the experiment.
        :param description: Optional description of the experiment.
        :return: Created experiment.
        """
        url = self._construct_url(self._base_url, API_VERSION, EXPERIMENTS) + '/'
        return self._handle_request(
            url=url, method=POST, json_data={NAME: name, DESCRIPTION: description}
        )

    def get_experiment(self, name: str) -> dict:
        """Get an experiment by its name.

        :param name: The name of the experiment.
        :return: Requested experiment.
        """
        url = self._construct_url(self._base_url, API_VERSION, EXPERIMENTS, name)
        return self._handle_request(url=url, method=GET)

    def list_experiments(self, page: int = 1, size: int = 50) -> Dict[str, Any]:
        """List every experiment.

        :param page: The page index.
        :param size: The number of experiments to retrieve.
        :return: Paginated results as a dict, where `items` are the `Experiment`s being
            retrieved.

        ..note:: `page` and `size` works together. For example, page 1 size 50 gives
            the 50 experiments in the first page. To get 50-100, set `page` as 2.
        ..note:: The maximum number for `size` per page is 100.
        """
        params = {'page': page, 'size': size}
        url = self._construct_url(self._base_url, API_VERSION, EXPERIMENTS) + '/'
        return self._handle_request(url=url, method=GET, params=params)

    def delete_experiment(self, name: str) -> Dict[str, Any]:
        """Delete an experiment given its name.

        :param name: The name of the experiment.
        :return: Experiment to be deleted.
        """
        url = self._construct_url(self._base_url, API_VERSION, EXPERIMENTS, name)
        return self._handle_request(url=url, method=DELETE)

    def delete_experiments(self) -> List[dict]:
        """Delete all experiments.

        :return: Experiments to be deleted.
        """
        url = self._construct_url(self._base_url, API_VERSION, EXPERIMENTS) + '/'
        return self._handle_request(url=url, method=DELETE)

    """ Run API """

    def get_run(self, experiment_name: str, run_name: str) -> dict:
        """Get a run by its name and experiment.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: Requested run.
        """
        url = self._construct_url(
            self._base_url, API_VERSION, EXPERIMENTS, experiment_name, RUNS, run_name
        )
        return self._handle_request(url=url, method=GET)

    def list_runs(
        self, experiment_name: Optional[str] = None, page: int = 50, size: int = 50
    ) -> Dict[str, Any]:
        """List all created runs inside a given experiment.

        If no experiment is specified, list runs for all available experiments.
        :param experiment_name: The name of the experiment.
        :param page: The page index.
        :param size: Number of runs to retrieve.
        :return: Paginated results as a dict, where `items` are the `Runs` being
            retrieved.

        ..note:: `page` and `size` works together. For example, page 1 size 50 gives
            the 50 runs in the first page. To get 50-100, set `page` as 2.
        ..note:: The maximum number for `size` per page is 100.
        """
        if not experiment_name:
            url = self._construct_url(self._base_url, API_VERSION, RUNS, RUNS)
        else:
            url = self._construct_url(
                self._base_url, API_VERSION, EXPERIMENTS, experiment_name, RUNS
            )
        params = {'page': page, 'size': size}
        return self._handle_request(url=url, method=GET, params=params)

    def delete_run(self, experiment_name: str, run_name: str) -> Dict[str, Any]:
        """Delete a run by its name and experiment.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: Deleted run.
        """
        url = self._construct_url(
            self._base_url, API_VERSION, EXPERIMENTS, experiment_name, RUNS, run_name
        )
        return self._handle_request(url=url, method=DELETE)

    def delete_runs(self, experiment_name: str) -> List[dict]:
        """Delete all runs inside a given experiment.

        :param experiment_name: The name of the experiment.
        :return: List of all deleted runs.
        """
        url = self._construct_url(
            self._base_url, API_VERSION, EXPERIMENTS, experiment_name, RUNS
        )
        return self._handle_request(url=url, method=DELETE)

    def get_run_status(self, experiment_name: str, run_name: str) -> dict:
        """Get a run status by its name and experiment.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: Run status.
        """
        url = self._construct_url(
            self._base_url,
            API_VERSION,
            EXPERIMENTS,
            experiment_name,
            RUNS,
            run_name,
            STATUS,
        )

        try:
            response = self._handle_request(url=url, method=GET, timeout=10)
        # In the event that the request times out, attempt the request one more time.
        except ReadTimeout:
            response = self._handle_request(url=url, method=GET, timeout=10)
        return response

    def get_run_logs(self, experiment_name: str, run_name: str) -> str:
        """Get a run logs by its name and experiment.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: Run logs.
        """
        url = self._construct_url(
            self._base_url,
            API_VERSION,
            EXPERIMENTS,
            experiment_name,
            RUNS,
            run_name,
            LOGS,
        )
        return self._handle_request(url=url, method=GET)

    def stream_run_logs(self, experiment_name: str, run_name: str) -> Iterator[str]:
        """Streaming log events to the client as ServerSentEvents.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :yield: A log entry.
        """
        url = self._construct_url(
            self._base_url,
            API_VERSION,
            EXPERIMENTS,
            experiment_name,
            RUNS,
            run_name,
            LOGSTREAM,
        )
        response = self._handle_request(url=url, method=GET, stream=True)
        for entry in response.iter_lines():
            if entry:
                decoded_message: str = entry.decode('utf-8', errors='ignore')
                sep_pos = decoded_message.find(': ')
                if sep_pos != -1:
                    msg_type, msg = (
                        decoded_message[:sep_pos],
                        decoded_message[sep_pos + 2 :],
                    )
                    if msg_type in ('data', 'event'):
                        yield msg

    def get_run_metrics(self, experiment_name: str, run_name: str) -> Dict[str, Any]:
        """Get evaluation metrics of a run by its name and experiment.
        This is only possible after the run has successfully finished.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: Dict of metrics before and after fine-tuning
        """
        url = self._construct_url(
            self._base_url,
            API_VERSION,
            EXPERIMENTS,
            experiment_name,
            RUNS,
            run_name,
            METRICS,
        )
        return self._handle_request(url=url, method=GET)

    def get_run_examples(self, experiment_name: str, run_name: str) -> Dict[str, Any]:
        """Get results of example queries of a run by the run's name and experiment.
        This is only possible after the run has successfully finished.

        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :return: Dict of results for example queries before and after fine-tuning.
        """
        url = self._construct_url(
            self._base_url,
            API_VERSION,
            EXPERIMENTS,
            experiment_name,
            RUNS,
            run_name,
            EXAMPLES,
        )
        return self._handle_request(url=url, method=GET)

    def create_run(
        self,
        experiment_name: str,
        run_name: str,
        run_config: dict,
        task: str,
        device: str,
        cpus: int,
        gpus: int,
    ) -> dict:
        """Create a run inside a given experiment.

        For optional parameters please visit our documentation (link).
        :param experiment_name: The name of the experiment.
        :param run_name: The name of the run.
        :param run_config: The run configuration.
        :param device: The device to use, either `cpu` or `gpu`.
        :param cpus: The number of CPUs to use.
        :param gpus: The number of GPUs to use.
        :return: Created run.
        """
        url = self._construct_url(
            self._base_url, API_VERSION, EXPERIMENTS, experiment_name, RUNS
        )
        return self._handle_request(
            url=url,
            method=POST,
            json_data={
                NAME: run_name,
                CONFIG: run_config,
                TASK: task,
                FINETUNER_VERSION: _finetuner_core_version,
                DEVICE: device,
                CPUS: cpus,
                GPUS: gpus,
            },
        )
