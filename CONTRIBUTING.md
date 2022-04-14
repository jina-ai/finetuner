# Contributing

## Dev Install

Finetuner requires your local `jina` as the latest master. It is the best if you have `jina` installed
via `pip install -e .`.

```bash
git clone https://github.com/jina-ai/finetuner.git
cd finetuner
# pip install -r requirements.txt (only required when you do not have jina locally) 
pip install -e .
```

## Install tests requirements


## Enable precommit hook

The codebase is enforced with Black style, please enable precommit hook.

```bash
pre-commit install
```