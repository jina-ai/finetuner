# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased] - 2022-MM-DD

### Added

- Setup the project structure. ([#385](https://github.com/jina-ai/finetuner/pull/385))

- Create experiment endpoints. ([#386](https://github.com/jina-ai/finetuner/pull/386))

- Create run endpoints. ([#387](https://github.com/jina-ai/finetuner/pull/387))

- Add Hubble authentication. ([#388](https://github.com/jina-ai/finetuner/pull/388))

- Add docs and netlify deployment. ([#392](https://github.com/jina-ai/finetuner/pull/392)) 

- Implement `Run`, `Experiment` and `Finetuner` classes on top of the base client. ([#391](https://github.com/jina-ai/finetuner/pull/391)) 

- Basic error handling. ([#394](https://github.com/jina-ai/finetuner/pull/394)) 

- Create a complete version of the run config. ([#395](https://github.com/jina-ai/finetuner/pull/395))

- Improve unit testing. ([#396](https://github.com/jina-ai/finetuner/pull/396))

- Implement getting run logs. ([#400](https://github.com/jina-ai/finetuner/pull/400))

- Add experiment-related methods to finetuner. ([#402](https://github.com/jina-ai/finetuner/pull/402))

### Removed

- Delete all unnecessary files from the previous project. ([#384](https://github.com/jina-ai/finetuner/pull/384))

### Changed

- Change logic behind artifact-id and return jsonified `dicts` instead of `requests.Response` objects. ([#390](https://github.com/jina-ai/finetuner/pull/390))

- Adapt getting run status. ([#400](https://github.com/jina-ai/finetuner/pull/400))

### Fixed

- Resolve CI tests. ([#398](https://github.com/jina-ai/finetuner/pull/398)) 

- Fix `download_artifact`. ([#401](https://github.com/jina-ai/finetuner/pull/401))
