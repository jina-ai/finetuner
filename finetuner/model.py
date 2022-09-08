from stubs.model import *  # noqa F401


def get_header() -> Tuple[str, ...]:
    """Get table header."""
    return 'name', 'task', 'output_dim', 'architecture', 'description'


def get_row(model_stub) -> Tuple[str, ...]:
    """Get table row."""
    return (
        model_stub.descriptor,
        model_stub.task,
        str(model_stub.output_shape[1]),
        model_stub.architecture,
        model_stub.description,
    )
