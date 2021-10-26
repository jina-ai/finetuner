import os
import tempfile
import threading
import webbrowser
from typing import Optional

import jina.helper
from jina import Flow
from jina.logging.predefined import default_logger

from .executor import FTExecutor, DataIterator
from ..helper import AnyDNN, DocumentArrayLike


def fit(
    embed_model: AnyDNN,
    train_data: DocumentArrayLike,
    clear_labels_on_start: bool = False,
    port_expose: Optional[int] = None,
    runtime_backend: str = 'thread',
    loss: str = 'CosineSiameseLoss',
    **kwargs,
) -> None:
    """Fit the model in an interactive UI.

    :param embed_model: The embedding model to fine-tune
    :param train_data: Data on which to train the model
    :param clear_labels_on_start: If set True, will remove all labeled data.
    :param port_expose: The port to expose.
    :param runtime_backend: The parallel backend of the runtime inside the Pea, either ``thread`` or ``process``.
    :param loss: Which loss to use in training. Supported
        losses are:
        - ``CosineSiameseLoss`` for Siamese network with cosine distance
        - ``EuclideanSiameseLoss`` for Siamese network with eculidean distance
        - ``CosineTripletLoss`` for Triplet network with cosine distance
        - ``EuclideanTripletLoss`` for Triplet network with eculidean distance
    :param kwargs: Additional keyword arguments.
    """
    dam_path = tempfile.mkdtemp()
    stop_event = threading.Event()

    class MyExecutor(FTExecutor):
        def get_embed_model(self):
            return embed_model

        def get_stop_event(self):
            return stop_event

    f = (
        Flow(
            protocol='http',
            port_expose=port_expose,
            prefetch=1,
            runtime_backend=runtime_backend,
        )
        .add(
            uses=DataIterator,
            uses_with={
                'dam_path': dam_path,
                'clear_labels_on_start': clear_labels_on_start,
            },
        )
        .add(
            uses=MyExecutor,
            uses_with={
                'dam_path': dam_path,
                'loss': loss,
            },
        )
    )

    f.expose_endpoint('/next')  #: for allowing client to fetch for the next batch
    f.expose_endpoint('/fit')  #: for signaling the backend to fit on the labeled data
    f.expose_endpoint('/feed')  #: for signaling the backend to fit on the labeled data
    f.expose_endpoint(
        '/save'
    )  #: for signaling the backend to save the current state of the model
    f.expose_endpoint('/terminate')  #: for terminating the flow from frontend

    def extend_rest_function(app):
        """Allow FastAPI frontend to serve finetuner UI as a static webpage"""
        from fastapi.staticfiles import StaticFiles

        p = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ui')
        app.mount('/finetuner', StaticFiles(directory=p, html=True), name='static2')
        return app

    jina.helper.extend_rest_interface = extend_rest_function

    global is_frontend_open
    is_frontend_open = False

    with f:

        def open_frontend_in_browser(req):
            global is_frontend_open
            if is_frontend_open:
                return
            url_html_path = f'http://localhost:{f.port_expose}/finetuner'
            try:
                webbrowser.open(url_html_path, new=2)
            except:
                pass  # intentional pass, browser support isn't cross-platform
            finally:
                default_logger.info(f'Finetuner is available at {url_html_path}')
                is_frontend_open = True

        # feed train data into the labeler flow
        f.post(
            '/feed',
            train_data,
            request_size=10,
            show_progress=True,
            on_done=open_frontend_in_browser,
        )
        f.block(stop_event=stop_event)
