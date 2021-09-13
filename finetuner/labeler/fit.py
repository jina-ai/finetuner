import os
import tempfile
import webbrowser
from collections import Iterable
from typing import Optional

import jina.helper
from jina import Flow, DocumentArray, DocumentArrayMemmap
from jina.logging.predefined import default_logger

from .executor import FTExecutor, DataIterator
from ..tuner.base import DocumentArrayLike, AnyDNN


def fit(
    embed_model: AnyDNN,
    head_layer: str,
    unlabeled_data: DocumentArrayLike,
    clear_labels_on_start: bool = False,
    port_expose: Optional[int] = None,
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

    class MyExecutor(FTExecutor):
        def get_embed_model(self):
            return embed_model

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
        .add(
            uses=MyExecutor,
            uses_with={
                'dam_path': dam_path,
                'head_layer': head_layer,
            },
        )
    )

    f.expose_endpoint('/next')
    f.expose_endpoint('/fit')

    def extend_rest_function(app):
        from fastapi.staticfiles import StaticFiles

        p = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ui')
        app.mount('/finetuner', StaticFiles(directory=p, html=True), name='static2')
        return app

    jina.helper.extend_rest_interface = extend_rest_function

    with f:
        f.logger.info('finetuner labeler is available at')
        url_html_path = f'http://localhost:{f.port_expose}/finetuner'
        try:
            webbrowser.open(url_html_path, new=2)
        except:
            pass  # intentional pass, browser support isn't cross-platform
        finally:
            default_logger.info(
                f'You should see a webpage opened in your browser, '
                f'if not you may open `{url_html_path}` manually.'
            )
        f.block()
