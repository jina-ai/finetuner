from dataclasses import dataclass
from typing import List, Union

from _finetuner.runner.stubs import model
from _finetuner.runner.stubs.model import *  # noqa F401
from _finetuner.runner.stubs.model import _EmbeddingModelStub


def get_header() -> Tuple[str, ...]:
    """Get table header."""
    return 'name', 'task', 'output_dim', 'architecture', 'description'


def get_row(model_stub) -> Tuple[str, ...]:
    """Get table row."""
    return (
        model_stub.display_name,
        model_stub.task,
        str(model_stub.output_shape[1]),
        model_stub.architecture,
        model_stub.description,
    )


def list_model_classes() -> Dict[str, ModelStubType]:
    rv = {}
    members = inspect.getmembers(model, inspect.isclass)
    parent_class = _EmbeddingModelStub
    for name, stub in members:
        if (
            name != 'MLPStub'
            and not name.startswith('_')
            and type(stub) != type
            and issubclass(stub, parent_class)
        ):
            rv[name] = stub
    return rv


@dataclass
class SynthesisModels:
    """Class specifying the models to be used in a data synthesis job.
    :param: relation_miner: The name of the model or list of models to use for
        relation mining.
    :param cross_encoder: The name of the model to use as the cross encoder
    """

    relation_miner: Union[str, List[str]]
    cross_encoder: str


synthesis_model_en = SynthesisModels(
    relation_miner='sbert-base-en',
    cross_encoder='crossencoder-base-en',
)
