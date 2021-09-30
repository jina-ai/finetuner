import os
import tempfile
import webbrowser
from collections import Iterable
from typing import Optional

import jina.helper
from jina import Flow, DocumentArray, DocumentArrayMemmap
from jina.logging.predefined import default_logger
from jina.logging.profile import TimeContext

from .executor import FTExecutor, DataIterator
from ..helper import AnyDNN, DocumentArrayLike


def fit(
    embed_model: AnyDNN,
    train_data: DocumentArrayLike,
    clear_labels_on_start: bool = False,
    port_expose: Optional[int] = None,
    runtime_backend: str = 'thread',
    head_layer: str = 'CosineLayer',
) -> None:

    with TimeContext('preparing data'):
        if callable(train_data):
            train_data = train_data()

        if isinstance(train_data, DocumentArray):
            dam_path = tempfile.mkdtemp()
            dam = DocumentArrayMemmap(dam_path)
            dam.extend(train_data)
        elif isinstance(train_data, DocumentArrayMemmap):
            dam_path = train_data.path
        elif isinstance(train_data, str):
            dam_path = train_data
        elif isinstance(train_data, Iterable):
            dam_path = tempfile.mkdtemp()
            dam = DocumentArrayMemmap(dam_path)
            dam.extend(train_data)
        else:
            raise TypeError(f'{train_data} is not supported')

    class MyExecutor(FTExecutor):
        def get_embed_model(self):
            return embed_model

    f = (
        Flow(protocol='http', port_expose=port_expose)
        .add(
            uses=DataIterator,
            uses_with={
                'dam_path': dam_path,
                'clear_labels_on_start': clear_labels_on_start,
            },
            runtime_backend=runtime_backend,
        )
        .add(
            uses=MyExecutor,
            uses_with={
                'dam_path': dam_path,
                'head_layer': head_layer,
            },
            runtime_backend=runtime_backend,  # eager-mode tf2 (M1-compiled) can not be run under `process` mode
        )
    )

    f.expose_endpoint('/next')  #: for allowing client to fetch for the next batch
    f.expose_endpoint('/fit')  #: for signaling the backend to fit on the labeled data

    def extend_rest_function(app):
        """Allow FastAPI frontend to serve finetuner UI as a static webpage"""
        from fastapi.staticfiles import StaticFiles

        p = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ui')
        app.mount('/finetuner', StaticFiles(directory=p, html=True), name='static2')
        return app

    jina.helper.extend_rest_interface = extend_rest_function

    with f:
        url_html_path = f'http://localhost:{f.port_expose}/finetuner'
        try:
            webbrowser.open(url_html_path, new=2)
        except:
            pass  # intentional pass, browser support isn't cross-platform
        finally:
            default_logger.info(f'Finetuner is available at {url_html_path}')
        f.block()
