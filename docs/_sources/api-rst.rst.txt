======================
:fab:`python` Python API
======================

This section includes the API documentation from the `Finetuner` codebase, as extracted from the `docstrings <https://peps.python.org/pep-0257/>`_ in the code.

:mod:`finetuner.__init__` - Finetuner
--------------------

.. currentmodule:: finetuner.__init__

.. autosummary::
   :nosignatures:
   :template: class.rst

   finetuner.login
   finetuner.describe_models
   finetuner.fit
   finetuner.list_callbacks
   finetuner.get_run
   finetuner.get_experiment
   finetuner.get_token
   finetuner.build_model
   finetuner.get_model
   finetuner.encode
   finetuner.list_runs
   finetuner.delete_run
   finetuner.delete_runs
   finetuner.create_experiment
   finetuner.list_experiments
   finetuner.delete_experiment
   finetuner.delete_experiments

:mod:`finetuner.run.Run` - Run
--------------------

.. currentmodule:: finetuner.run.Run

.. autosummary::
   :nosignatures:
   :template: class.rst

   finetuner.run.Run.name
   finetuner.run.Run.config
   finetuner.run.Run.status
   finetuner.run.Run.logs
   finetuner.run.Run.stream_logs
   finetuner.run.Run.save_artifact
   finetuner.run.Run.artifact_id


:mod:`finetuner.experiment.Experiment` - Experiment
--------------------

.. currentmodule:: finetuner.experiment.Experiment

.. autosummary::
   :nosignatures:
   :template: class.rst

   finetuner.experiment.Experiment.name
   finetuner.experiment.Experiment.create_run
   finetuner.experiment.Experiment.get_run
   finetuner.experiment.Experiment.list_runs
   finetuner.experiment.Experiment.delete_run
   finetuner.experiment.Experiment.delete_runs

