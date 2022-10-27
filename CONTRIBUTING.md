# Contributing


## Setup

### Install dev requirements

```bash
make install-dev
```

### Install finetuner

```bash
make install
```

### Enable precommit hook

To automatically ensure formatting with `black`, import sorting with `isort` and linting
with `flake8`, you can install the pre-commit hooks 

```bash
make pre-commit
```


## Making a PR

### Open an issue

Each PR should reference an open issue, and this issue should be linked to your PR.

### Running tests locally

To run tests locally, all you need to do is

```bash
make test
```

### Adding an entry to the changelog

Make an entry in [CHANGELOG.md](https://github.com/jina-ai/finetuner/blob/main/CHANGELOG.md),
adding it to the `Unreleased` section (and the appropriate subsection), which should contain a
short description of what you have done in the PR, as well as the PR's number, e.g.

```
- Add `NTXentLoss` loss class for supervised learning ([#24](https://github.com/jina-ai/finetuner.fit/pull/24))
```

To avoid merge conflicts when multiple people are simultaneously working on new features, make sure there
is **an empty line above and below the entry**.

## Update notebooks

We have three Google Colab embedded inside the documentation:

- [text-to-text with bert](https://colab.research.google.com/drive/1Ui3Gw3ZL785I7AuzlHv3I0-jTvFFxJ4_?usp=sharing)
- [image-to-image with resnet](https://colab.research.google.com/drive/1QuUTy3iVR-kTPljkwplKYaJ-NTCgPEc_?usp=sharing)
- [text-to-iamge with clip](https://colab.research.google.com/drive/1yKnmy2Qotrh3OhgwWRsMWPFwOSAecBxg?usp=sharing)

To update code in colab:

1. Update code in the Google Colab.
2. Download into `docs/notebooks/` folder.
3. cd into `docs` folder, run `make notebook` and run `make dirhtml` to see output locally.

Only members of the team have the permissions to modify the notebook.

## Releases

To make a release, follow these steps, in order.

### Update CHANGELOG.md

In `CHANGELOG.md`, rename the top `Unreleased` entry with the with the version number (`X.Y.Z`), and enter the current date.

Then, add a new empty `Unreleased` section on top of it - this is where the changes for the next version will accumulate.

### Tag the commit on `main` branch

In your repository, check out the `main` branch, and tag it with the appropriate version - it should match the one in `finetuner/__init__.py`!
If it does not, change it there first.

To tag the head commit in `main` branch, and then push this to remote, do the following steps
(you can also do this automatially by creating a release on GitHub)

```bash
git checkout main
git tag vX.Y.Z
git push --tags
```

At this point the new version is officially released. At this point any automated actions connected
to release would have been run.

### Change version in `finetuner/__init__.py`

Since now the `main` branch corresponds to the new development version, we need to change the version
in `finetuner/__init__.py` to reflect that. So you should increment the version in that file.
