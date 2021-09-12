from typing import Optional, Type

from jina import Flow

from .executor import FTExecutor, DataIterator


def start_labeler_flow(
    embed_executor: Type[FTExecutor],
    dam_path: str,
    labeled_dam_path: str,
    clear_labels_on_start: bool = False,
    port_expose: Optional[int] = None,
    **kwargs
):
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
