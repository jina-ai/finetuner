from _finetuner.runner.stubs import model
from _finetuner.runner.stubs.model import *  # noqa F401
from _finetuner.runner.stubs.model import (
    _CrossEncoderStub,
    _EmbeddingModelStub,
    _TextTransformerStub,
)

from finetuner.constants import CROSS_ENCODING, EMBEDDING, RELATION_MINING


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


def list_model_classes(model_type: str = EMBEDDING) -> Dict[str, ModelStubType]:
    rv = {}
    members = inspect.getmembers(model, inspect.isclass)
    if model_type == EMBEDDING:
        parent_class = _EmbeddingModelStub
    elif model_type == CROSS_ENCODING:
        parent_class = _CrossEncoderStub
    elif model_type == RELATION_MINING:
        parent_class = _TextTransformerStub
    for name, stub in members:
        if (
            name != 'MLPStub'
            and not name.startswith('_')
            and type(stub) != type
            and issubclass(stub, parent_class)
        ):
            rv[name] = stub
    return rv
