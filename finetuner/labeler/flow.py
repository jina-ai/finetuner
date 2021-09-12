import tempfile
from collections import Iterable
from typing import Optional, Type

from jina import Flow, DocumentArray, DocumentArrayMemmap

from .executor import FTExecutor, DataIterator
from ..tuner.base import DocumentArrayLike


def start_label_flow(
    embed_executor: Type[FTExecutor],
    unlabeled_data: DocumentArrayLike,
    clear_labels_on_start: bool = False,
    port_expose: Optional[int] = None,
    **kwargs,
):
    if callable(unlabeled_data):
        unlabeled_data = unlabeled_data()

    if isinstance(unlabeled_data, DocumentArray):
        dam_path = tempfile.mkdtemp()
        dam = DocumentArrayMemmap(dam_path)
        dam.extend(unlabeled_data)
    elif isinstance(unlabeled_data, DocumentArrayMemmap):
        dam_path = unlabeled_data._path
    elif isinstance(unlabeled_data, str):
        dam_path = unlabeled_data
    elif isinstance(unlabeled_data, Iterable):
        dam_path = tempfile.mkdtemp()
        dam = DocumentArrayMemmap(dam_path)
        dam.extend(unlabeled_data)
    else:
        raise TypeError(f'{unlabeled_data} is not supported')

    labeled_dam_path = f'{dam_path}/labeled'

    f = (
        Flow(protocol='http', port_expose=port_expose, cors=True)
        .add(
            uses=DataIterator,
            uses_with={
                'dam_path': dam_path,
                'labeled_dam_path': labeled_dam_path,
                'clear_labels_on_start': clear_labels_on_start,
            },
        )
        .add(uses=embed_executor, uses_with={'dam_path': dam_path})
    )

    f.expose_endpoint('/next')
    f.expose_endpoint('/fit')
    with f:
        f.block()
