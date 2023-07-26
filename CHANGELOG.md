# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased] - 2023-MM-DD

### Added

### Removed

### Changed

### Fixed

### Docs


## [0.8.1] - 2023-07-26

### Added

### Removed

### Changed

### Fixed

### Docs

- Add tiny model and citation to Readme and docs. ([#763](https://github.com/jina-ai/finetuner/pull/763))

- Fix huggingface link of jina embeddings. ([#761](https://github.com/jina-ai/finetuner/pull/761))

- Remove redundant text in jina embedding page. ([#762](https://github.com/jina-ai/finetuner/pull/762))


## [0.8.0] - 2023-07-13

### Added

- Add jina embeddings suit. ([#757](https://github.com/jina-ai/finetuner/pull/757))

- Add `cos_sim` helper to finetuner. ([#757](https://github.com/jina-ai/finetuner/pull/757))

### Removed

### Changed

- Finetuner always install torch and other dependencies. ([#757](https://github.com/jina-ai/finetuner/pull/757))

### Fixed

### Docs

- Add Jina embeddings documentation page. ([#759](https://github.com/jina-ai/finetuner/pull/759))


## [0.7.8] - 2023-06-08

### Added

- Support loading models from Jina's huggingface site. ([#751](https://github.com/jina-ai/finetuner/pull/751))

- Add multilingual model for training data generation job. ([#750](https://github.com/jina-ai/finetuner/pull/750))

### Removed

### Changed

- Bump black, flake8 and isort ([#747](https://github.com/jina-ai/finetuner/pull/747))

- Increase the default `num_relations` from 3 to 10 for data synthesis job. ([#750](https://github.com/jina-ai/finetuner/pull/750))

### Fixed

- Fix create synthesis run not accepting `DocumentArray` as input type. ([#748](https://github.com/jina-ai/finetuner/pull/748))

### Docs

- Update data synthesis tutorial include english and multilingual model. ([#750](https://github.com/jina-ai/finetuner/pull/750))


## [0.7.7] - 2023-05-24

### Added

- Add support for data generation jobs. ([#715](https://github.com/jina-ai/finetuner/pull/715))

### Removed

### Changed

- Import `Document`, `DocumentArray` from finetuner. ([#720](https://github.com/jina-ai/finetuner/pull/720))

### Fixed

### Docs

- Add documentation on using `Document` and `DocumentArray` from docarray v1. ([#720](https://github.com/jina-ai/finetuner/pull/720))

- Add notebook on data generation. ([#745](https://github.com/jina-ai/finetuner/pull/745))


## [0.7.6] - 2023-04-18

### Added

### Removed

### Changed

- Install finetuner from source code instead of using a pip package [#719](https://github.com/jina-ai/finetuner/pull/719).

### Fixed

- Downgrade docarray version [#719](https://github.com/jina-ai/finetuner/pull/719).

### Docs


## [0.7.5] - 2023-04-14

### Added

- Add support for python 3.10. ([#704](https://github.com/jina-ai/finetuner/pull/704))

- Add error message when `experiment` is `None` when creating a run. ([#708](https://github.com/jina-ai/finetuner/pull/708))

### Removed

### Changed

- Use correct schema and model stub for fine-tuning job rather than generation job. ([#711](https://github.com/jina-ai/finetuner/pull/711))

### Fixed

- Fix error caused by pods restarting. ([#709](https://github.com/jina-ai/finetuner/pull/709))

### Docs


## [0.7.4] - 2023-03-29

### Added

- Support pair-wise score document construction from CSV. ([#696](https://github.com/jina-ai/finetuner/pull/696))

### Removed

### Changed

- Refactor `load_finetuning_dataset` into CSV handlers. ([#696](https://github.com/jina-ai/finetuner/pull/696))

- Unify all models names into `name-size-lang` format. ([#700](https://github.com/jina-ai/finetuner/pull/700))

- Do not download pre-trained weights when user downloads the artifact. ([#706](https://github.com/jina-ai/finetuner/pull/706))

### Fixed

### Docs

- Add query-document score into CSV data preparation. ([#697](https://github.com/jina-ai/finetuner/pull/697))

- Add `CosineSimilarityLoss` into advanced loss section. ([#697](https://github.com/jina-ai/finetuner/pull/697))

- Add layer-wise learning rate decay into advanced configurations. ([#697](https://github.com/jina-ai/finetuner/pull/697))


## [0.7.3] - 2023-03-16

### Added

- Add support for batch size scaling. ([#691](https://github.com/jina-ai/finetuner/pull/691))

- Add functions to retrieve evaluation metrics and example results. ([#687](https://github.com/jina-ai/finetuner/pull/687))

### Removed

### Changed

- Remove unit test and integration test from CD. ([#686](https://github.com/jina-ai/finetuner/pull/686))

### Fixed

### Docs


## [0.7.2] - 2023-03-02

### Added

- Add support for learning rate schedulers. ([#679](https://github.com/jina-ai/finetuner/pull/679))

### Removed

### Changed

- Remove duplicated documents when parsing unlabeled CSV files. ([#678](https://github.com/jina-ai/finetuner/pull/678))

- The `scheduler_step` options is now part of `scheduler_options`. ([#679](https://github.com/jina-ai/finetuner/pull/679))

### Fixed

### Docs

- Update documentation on creating training data. ([#678](https://github.com/jina-ai/finetuner/pull/678))

- Add notebook to demonstrate use of `ArcFaceLoss`. ([#680](https://github.com/jina-ai/finetuner/pull/680))

- Add section on GeM pooling to advanced topics. ([#684](https://github.com/jina-ai/finetuner/pull/684))


## [0.7.1] - 2023-02-15

### Added

- Add support for new loss and pooling options to the `finetuner.fit` method. ([#664](https://github.com/jina-ai/finetuner/pull/664))

- Add folder for example CSV files. ([#663](https://github.com/jina-ai/finetuner/pull/663))

- Add communication between remote-ci job and the pr that triggered it. ([#642](https://github.com/jina-ai/finetuner/pull/642))

- Support continuing training from an artifact of a previous run. ([#668](https://github.com/jina-ai/finetuner/pull/668))

### Removed

### Changed

- Use github token provided by dispatcher when running remote-ci. ([#640](https://github.com/jina-ai/finetuner/pull/640))

### Fixed

- Use python 3.8 in Github actions. ([#659](https://github.com/jina-ai/finetuner/pull/659))

- Add proper CSV file for image-image case. ([#667](https://github.com/jina-ai/finetuner/pull/667))

- Fix problems with login function in notebooks by bumping hubble version. ([#672](https://github.com/jina-ai/finetuner/pull/672))

- Fix URL construction. ([#672](https://github.com/jina-ai/finetuner/pull/672)) 

### Docs

- Add page on loss and pooling to `advanced-topics`. ([#664](https://github.com/jina-ai/finetuner/pull/664))

- Remove ResNet backbone support for clip fine-tuning. ([#662](https://github.com/jina-ai/finetuner/pull/662))

- Add efficientnet b7 as a new image to image search backbone. ([#662](https://github.com/jina-ai/finetuner/pull/662))

- Fix typos, duplicate paragraphs, and wrong formulations. ([#666](https://github.com/jina-ai/finetuner/pull/666)) 

- Add list of articles to README and docs. ([#669](https://github.com/jina-ai/finetuner/pull/669))

- Removed section on GeM pooling from advanced topics. ([#676](https://github.com/jina-ai/finetuner/pull/676))


## [0.7.0] - 2023-01-18

### Added

- Add `val_split` parameter to `fit` function. ([#624](https://github.com/jina-ai/finetuner/pull/624))

- Add `core-ci` workflow to remotely run the ci of finetuner-core. ([#628](https://github.com/jina-ai/finetuner/pull/628))

- Add support for 3d meshes to `build_finetuning_dataset`. ([#638](https://github.com/jina-ai/finetuner/pull/638))

### Removed

- Remove `cpu` parameter from `create_run` function. ([#631](https://github.com/jina-ai/finetuner/pull/631))

- Remove `notebook_login` function. ([#631](https://github.com/jina-ai/finetuner/pull/631))

- Remove support for python 3.7 ([#653](https://github.com/jina-ai/finetuner/pull/653))

### Changed

- Adjust Finetuner based on API changes for Jina AI Cloud. ([#637](https://github.com/jina-ai/finetuner/pull/637))

- Change default `experiment_name` from current working dir to `default`. ([#637](https://github.com/jina-ai/finetuner/pull/637))

- Use github token provided by dispatcher when running remote-ci. ([#640](https://github.com/jina-ai/finetuner/pull/640))

### Fixed

- Correctly infer the type of models created using `get_model` in the `build_encoding_dataset` function. ([#623](https://github.com/jina-ai/finetuner/pull/623))

### Docs

- Add before and after section to the example notebooks. ([#622](https://github.com/jina-ai/finetuner/pull/622))

- Align text-to-image notebook with its corresponding markdown file. ([#621](https://github.com/jina-ai/finetuner/pull/621))

- Change hint in notebooks to use `load_uri_to_blob` instead of `load_uri_to_image_tensor`. ([#625](https://github.com/jina-ai/finetuner/pull/625))

- Copyedit `README.md`, changes to language but not contents. ([#635](https://github.com/jina-ai/finetuner/pull/635))

- Add multilingual clip colab to readme. ([#620](https://github.com/jina-ai/finetuner/pull/620))

- Add tutorial for mesh-to-mesh search. ([#638](https://github.com/jina-ai/finetuner/pull/638))

- Add documentation for PointNet++ model and handling 3D mesh dataset. ([#638](https://github.com/jina-ai/finetuner/pull/638))

- Add `finetuner` namespace to artifact names in the documentation. ([#649](https://github.com/jina-ai/finetuner/pull/649))

- Rewrite M-CLIP notebook to use German fashion dataset. ([#643](https://github.com/jina-ai/finetuner/pull/643))

- New advanced topics section. ([#643](https://github.com/jina-ai/finetuner/pull/643))

- Improve developer reference. ([#643](https://github.com/jina-ai/finetuner/pull/643))

- Improve walkthrough sections. ([#643](https://github.com/jina-ai/finetuner/pull/643))

- Add hints to escape common to prepare csv training data. ([#655](https://github.com/jina-ai/finetuner/pull/655))


## [0.6.7] - 2022-11-25

### Added

- Allow user to control `num_items_per_class` to sample to each batch. ([#614](https://github.com/jina-ai/finetuner/pull/614))

### Removed

### Changed

- Update commons and stubs versions. ([#618](https://github.com/jina-ai/finetuner/pull/618))

### Fixed

- Valid configuration of `num_items_per_class`. ([#618](https://github.com/jina-ai/finetuner/pull/618))

### Docs

- Add notebook for multilingual CLIP models. ([#611](https://github.com/jina-ai/finetuner/pull/611))

- Improve `describe_models` with `task` to better organize list of backbones. ([#610](https://github.com/jina-ai/finetuner/pull/610))

- Add documentation on using the evaluation callback for CLIP (multiple models). ([#615](https://github.com/jina-ai/finetuner/pull/615))

- Ignore `callback` module in apidoc. ([#614](https://github.com/jina-ai/finetuner/pull/614))


## [0.6.6] - 2022-11-24

This release was broken and was deleted.

### Added

### Removed

### Changed

### Fixed

### Docs


## [0.6.5] - 2022-11-10

### Added

- Add support for CSV files to the `EvaluationCallback`. ([#608](https://github.com/jina-ai/finetuner/pull/608))

- Add support for CSV files to the `fit` function. ([#592](https://github.com/jina-ai/finetuner/pull/592))

- Add support for lists to the `encode` function. [#598](https://github.com/jina-ai/finetuner/pull/598)

- Allow user to publish public artifact. [#602](https://github.com/jina-ai/finetuner/pull/602)

### Removed

- Remove `connect` function. ([#596](https://github.com/jina-ai/finetuner/pull/596))

### Changed

- Enhance documentation of login functionalities. ([#596](https://github.com/jina-ai/finetuner/pull/596))

- Deprecate `notebook_login` function with `login(interactive=True)`. ([#594](https://github.com/jina-ai/finetuner/pull/594))

### Fixed

- Correctly use `eval_data` in the `create_run` function ([#603](https://github.com/jina-ai/finetuner/pull/603))

- Fix links to functions in the documentation. ([#596](https://github.com/jina-ai/finetuner/pull/596))

### Docs

- Improve documentation on csv reading and run monitoring section. [#601](https://github.com/jina-ai/finetuner/pull/601)

- Add documentation for `WandBLogger`. [#600](https://github.com/jina-ai/finetuner/pull/600)

- Change datasets and hyperparameters for ResNet experiment. ([#599](https://github.com/jina-ai/finetuner/pull/599))

- Use `login` instead of `notebook_login` in examples. ([#605](https://github.com/jina-ai/finetuner/pull/605))


## [0.6.4] - 2022-10-27

### Added

- Add `build_model` function to create zero-shot models. ([#584](https://github.com/jina-ai/finetuner/pull/584))

- Use latest Hubble with `notebook_login` support. ([#576](https://github.com/jina-ai/finetuner/pull/576))

### Removed

### Changed

- Use the run config model from `finetuner-stubs` to create the run config. ([#579](https://github.com/jina-ai/finetuner/pull/579))

- Use `device` parameter to replace `cpu` to align with docarray. ([#577](https://github.com/jina-ai/finetuner/pull/577))

- Update the open clip model names in the table of the backbones. ([#580](https://github.com/jina-ai/finetuner/pull/580))

- Show progress while encode batches of documents. ([#586](https://github.com/jina-ai/finetuner/pull/586))

- Change `device` as an optional parameter when calling `get_model`. ([#586](https://github.com/jina-ai/finetuner/pull/586))

### Fixed

### Docs

- Fix training data name in totally looks like example. ([#576](https://github.com/jina-ai/finetuner/pull/576))

- Embed three tasks as three Google Colab notebooks in documentation. ([#583](https://github.com/jina-ai/finetuner/pull/583))

- Unify documentation related to cloud storage as Jina AI Cloud. ([#582](https://github.com/jina-ai/finetuner/pull/582))

- Replace `hub.jina.ai` with `cloud.jina.ai`. ([#587](https://github.com/jina-ai/finetuner/pull/587))


## [0.6.3] - 2022-10-13

### Added

- Support advanced CLIP fine-tuning with WiSE-FT. ([#571](https://github.com/jina-ai/finetuner/pull/571))

### Removed

### Changed

- Change CLIP fine-tuning example in the documentation. ([#569](https://github.com/jina-ai/finetuner/pull/569))

### Fixed

- Bump flake8 to `5.0.4`. ([#568](https://github.com/jina-ai/finetuner/pull/568))

### Docs

- Add documentation for callbacks. ([#567](https://github.com/jina-ai/finetuner/pull/567))


## [0.6.2] - 2022-09-29

### Added

- Support inference with torch models. ([#560](https://github.com/jina-ai/finetuner/pull/560))

### Removed

### Changed

### Fixed

- Freeze hubble client to `0.17.0`. ([#556](https://github.com/jina-ai/finetuner/pull/556))

### Docs

- Fix template html css. ([#556](https://github.com/jina-ai/finetuner/pull/556))


## [0.6.1] - 2022-09-27

### Added

- Add `finetuner_version` equal to the stubs version in the create run request. ([#552](https://github.com/jina-ai/finetuner/pull/552))

### Removed

- Improve display of stream log messages. ([#549](https://github.com/jina-ai/finetuner/pull/549))

### Changed

- Bump hubble client version. ([#546](https://github.com/jina-ai/finetuner/pull/546))

### Fixed

- Preserve request headers in redirects to the same domain. ([#552](https://github.com/jina-ai/finetuner/pull/552))

### Docs

- Improve example and install documentation. ([#534](https://github.com/jina-ai/finetuner/pull/534))

- Update finetuner executor version in docs. ([#543](https://github.com/jina-ai/finetuner/pull/543))


## [0.6.0] - 2022-09-09

### Added

- Add `get_model` and `encode` method to encode docarray. ([#522](https://github.com/jina-ai/finetuner/pull/522))

- Add connect function to package. ([#532](https://github.com/jina-ai/finetuner/pull/532))

### Removed

### Changed

- Incorporate `commons` and `stubs` to use shared components. ([#522](https://github.com/jina-ai/finetuner/pull/522))

- Improve usability of `stream_logs`. ([#522](https://github.com/jina-ai/finetuner/pull/522))

- Improve `describe_models` with open-clip models. ([#528](https://github.com/jina-ai/finetuner/pull/528))

- Use stream logging in the README example. ([#532](https://github.com/jina-ai/finetuner/pull/532))

### Fixed

- Print logs before run status is `STARTED`. ([#531](https://github.com/jina-ai/finetuner/pull/531))

### Docs

- Add inference session in examples. ([#529](https://github.com/jina-ai/finetuner/pull/529))


## [0.5.2] - 2022-08-31

### Added

- Description of get_model and encode function. ([#526](https://github.com/jina-ai/finetuner/pull/526))

- Enable wandb callback. ([#494](https://github.com/jina-ai/finetuner/pull/494))

- Support log streaming in finetuner client. ([#504](https://github.com/jina-ai/finetuner/pull/504))

- Support optimizer and miner options. [#517](https://github.com/jina-ai/finetuner/pull/517)

### Removed

### Changed

- Mark fit as login required. ([#494](https://github.com/jina-ai/finetuner/pull/494))

- Improve documentation according to remarks we received. ([524](https://github.com/jina-ai/finetuner/pull/524))

### Fixed

- Replace the artifact name from dot to dash. ([#519](https://github.com/jina-ai/finetuner/pull/519))

- Create client automatically if user is already logged in. ([#527](https://github.com/jina-ai/finetuner/pull/527))

### Docs

- Fix google analytics Id for docs. ([#499](https://github.com/jina-ai/finetuner/pull/499))

- Update sphinx-markdown-table to v0.0.16 to get. [this fix](https://github.com/ryanfox/sphinx-markdown-tables/pull/37) ([#499](https://github.com/jina-ai/finetuner/pull/499))

- Place install instructions in the documentation more prominent. ([#518](https://github.com/jina-ai/finetuner/pull/518))


## [0.5.1] - 2022-07-15

### Added

- Add artifact id and token interface to improve usability. ([#485](https://github.com/jina-ai/finetuner/pull/485))

### Removed

### Changed

- `save_artifact` should show progress while downloading. ([#483](https://github.com/jina-ai/finetuner/pull/483))

- Give more flexibility on dependency versions. ([#483](https://github.com/jina-ai/finetuner/pull/483))

- Bump `jina-hubble-sdk` to 0.8.1. ([#488](https://github.com/jina-ai/finetuner/pull/488))

- Improve integration section in documentation. ([#492](https://github.com/jina-ai/finetuner/pull/492))

- Bump `docarray` to 0.13.31. ([#492](https://github.com/jina-ai/finetuner/pull/492))

### Fixed

- Use `uri` to represent image content in documentation creating training data code snippet. ([#484](https://github.com/jina-ai/finetuner/pull/484))

- Remove out-dated CLIP-specific documentation. ([#491](https://github.com/jina-ai/finetuner/pull/491))


## [0.5.0] - 2022-06-30

### Added

- Docs 0.4.1 backup. ([#462](https://github.com/jina-ai/finetuner/pull/462))

- Add Jina integration section in the docs. ([#467](https://github.com/jina-ai/finetuner/pull/467))

- Add CD back with semantic release. ([#472](https://github.com/jina-ai/finetuner/pull/472))

### Removed

### Changed

- Refactor the guide for image to image search. ([#458](https://github.com/jina-ai/finetuner/pull/458))

- Refactor the guide for text to image search. ([#459](https://github.com/jina-ai/finetuner/pull/459))

- Refactor the default hyper-params and docstring format. ([#465](https://github.com/jina-ai/finetuner/pull/465))

- Various updates on style, how-to and templates. ([#462](https://github.com/jina-ai/finetuner/pull/462))

- Remove time column from Readme table. ([#468](https://github.com/jina-ai/finetuner/pull/468))

- Change release trigger to push to `main` branch. ([#478](https://github.com/jina-ai/finetuner/pull/478))

### Fixed

- Use finetuner docs links in docs instead of netlify. ([#475](https://github.com/jina-ai/finetuner/pull/475))

- Use twine pypi release. ([#480](https://github.com/jina-ai/finetuner/pull/480))

- Fix blocked success-all-tests in CI. ([#482](https://github.com/jina-ai/finetuner/pull/482))

- Fix documentation render in the login page. ([#482](https://github.com/jina-ai/finetuner/pull/482))


## 0.2.2 - 2022-06-16

### Added

### Removed

- Remove `path` and `dotenv` as dependencies. ([#444](https://github.com/jina-ai/finetuner/pull/444))

### Changed

- Change default registry to prod for api and hubble. ([#447](https://github.com/jina-ai/finetuner/pull/447))

- Polish the documentation structure and references. ([#460](https://github.com/jina-ai/finetuner/pull/460))

- Update README.md with latest developments. ([#448](https://github.com/jina-ai/finetuner/pull/448))

### Fixed


## 0.2.1 - 2022-06-13

### Added

### Removed

### Changed

### Fixed

- docs: fix link references and missing images. ([#439](https://github.com/jina-ai/finetuner/pull/439))

- fix: send another request when redirect detected. ([#441](https://github.com/jina-ai/finetuner/pull/441))


## 0.2.0 - 2022-06-09

### Added

- Add default values for finetuner `HOST` and `JINA_HUBBLE_REGISTRY`. ([#410](https://github.com/jina-ai/finetuner/pull/410))

- Expose arguments `cpu` and `num_workers` in `finetuner.fit`. ([#411](https://github.com/jina-ai/finetuner/pull/411))

- Add documentation structure and how it works section. ([#412](https://github.com/jina-ai/finetuner/pull/412))

- Support passing callbacks to the run configuration. ([#415](https://github.com/jina-ai/finetuner/pull/415))

- Add documentation step by step from install to create training data. ([#416](https://github.com/jina-ai/finetuner/pull/416))

- Add support for `EvaluationCallback`. ([#422](https://github.com/jina-ai/finetuner/pull/422))

- Docs add developer reference, Jina ecosystem and style fix. ([#423](https://github.com/jina-ai/finetuner/pull/423))

- Add support for MLP model. ([#428](https://github.com/jina-ai/finetuner/pull/428))

- Add method `list_models` that returns the available model names. ([#428](https://github.com/jina-ai/finetuner/pull/428))

- Organize supported model to model stubs under `finetuner.models`. ([#428](https://github.com/jina-ai/finetuner/pull/428))

- Add a guide for image-to-image retrieval. ([#430](https://github.com/jina-ai/finetuner/pull/430))

- Add a guide for text to image fine-tuning with `CLIP`. ([#433](https://github.com/jina-ai/finetuner/pull/433))

- Add template for guides in docs. ([#437](https://github.com/jina-ai/finetuner/pull/437))

- Add text to text with Bert guide to docs. ([#426](https://github.com/jina-ai/finetuner/pull/426))


### Removed

### Changed

- Bump `docarray` to `v0.13.17`. ([#411](https://github.com/jina-ai/finetuner/pull/411))

- Guide user to choose models in `list_models`. ([#419](https://github.com/jina-ai/finetuner/pull/419))

- Expose run methods. ([#425](https://github.com/jina-ai/finetuner/pull/425))

- Rename `list_models` to `describe_models`. ([#428](https://github.com/jina-ai/finetuner/pull/428))

- Rename `finetuner.callbacks` to `finetuner.callback` to avoid name collision in `__init__.py`. ([#428](https://github.com/jina-ai/finetuner/pull/428))

### Fixed

- Enable saving models for clip without overwriting. ([#432](https://github.com/jina-ai/finetuner/pull/432))

- Fix flaky integration test. ([#436](https://github.com/jina-ai/finetuner/pull/436))

## [0.1.0] - 2022-05-23

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

- Add CD step for PyPI release. ([#403](https://github.com/jina-ai/finetuner/pull/403))

### Removed

- Delete all unnecessary files from the previous project. ([#384](https://github.com/jina-ai/finetuner/pull/384))

### Changed

- Change logic behind artifact-id and return jsonified `dicts` instead of `requests.Response` objects. ([#390](https://github.com/jina-ai/finetuner/pull/390))

- Adapt getting run status. ([#400](https://github.com/jina-ai/finetuner/pull/400))

### Fixed

- Resolve CI tests. ([#398](https://github.com/jina-ai/finetuner/pull/398)) 

- Fix `download_artifact`. ([#401](https://github.com/jina-ai/finetuner/pull/401))
