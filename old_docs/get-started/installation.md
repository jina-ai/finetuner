(install-finetuner)=
# {octicon}`desktop-download` Installation

![PyPI](https://img.shields.io/pypi/v/finetuner?color=%23ffffff&label=%20) is the latest version.

Make sure you have `Python 3.8+` installed on Linux/Mac/Windows:

```bash
pip install -U finetuner
```

If you want to encode your data locally with the {meth}`~finetuner.encode` function, you need to install `"finetuner[full]"`.
In this case, some extra dependencies are installed which are necessary to do the inference, e.g., torch, torchvision, and open clip:

```bash
pip install "finetuner[full]"
```

To check your installation run:
```bash
pip show finetuner
```