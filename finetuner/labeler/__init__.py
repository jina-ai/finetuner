import os
import tempfile
import webbrowser
from typing import Optional, Dict

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
        head_layer: str = 'CosineLayer',
        model_definition_file_path: Optional[str] = None,
        extra_kwargs_model_init: Optional[Dict] = None,
        **kwargs,
) -> None:
    from ..helper import get_framework
    dam_path = tempfile.mkdtemp()
    checkpoint_path = tempfile.mkdtemp()
    embedding_model_cls = embed_model.__cls__.__name__
    framework = get_framework(embed_model)
    if framework == 'keras':
        embed_model.save_weights(checkpoint_path)
    elif framework == 'torch':
        import torch
        torch.save(embed_model.state_dict(), checkpoint_path)
    elif framework == 'paddle':
        raise Exception(' Did not find good documentation')

    class MyExecutor(FTExecutor):

        def __init__(self,
                     framework: Optional[str] = None,
                     embedding_model_cls: Optional[str] = None,
                     checkpoint_path: Optional[str] = None,
                     model_definition_file_path: Optional[str] = None,
                     extra_kwargs_model_init: Optional[Dict] = None,
                     *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._framework = framework
            self._embedding_model_cls = embedding_model_cls
            self._model_definition_file_path = model_definition_file_path
            self._checkpoint_path = checkpoint_path
            self._extra_kwargs_model_init = extra_kwargs_model_init

        def _get_embed_model_torch(self):
            import importlib
            import types
            import torch
            loader = importlib.machinery.SourceFileLoader(
                '__imported_module__', self.model_definition_file
            )
            mod = types.ModuleType(loader.name)
            loader.exec_module(mod)
            model = getattr(mod, self.model_class_name)(**self._extra_kwargs_model_init)
            model.load_state_dict(torch.load(self.model_state_dict_path))
            return model

        def _get_embed_model_paddle(self):
            pass

        def _get_embed_model_keras(self):
            import importlib
            import types
            from tensorflow import keras

            loader = importlib.machinery.SourceFileLoader(
                '__imported_module__', self.model_definition_file
            )
            mod = types.ModuleType(loader.name)
            loader.exec_module(mod)
            model = getattr(mod, self.model_class_name)(**self._extra_kwargs_model_init)
            model.load_weights(self.model_state_dict_path)
            return model

        def get_embed_model(self):
            if self._framework == 'torch':
                return self._get_embed_model_torch()
            elif self._framework == 'paddle':
                return self._get_embed_model_paddle()
            elif self._framework == 'keras':
                return self._get_embed_model_keras()

    f = (
        Flow(
            protocol='http',
            port_expose=port_expose,
            prefetch=1,
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
                'head_layer': head_layer,
                'framework': framework,
                'checkpoint_path': checkpoint_path,
                'embedding_model_cls': embedding_model_cls,
                'model_definition_file_path': model_definition_file_path,
                'extra_kwargs_model_init': extra_kwargs_model_init
            },
        )
    )

    f.expose_endpoint('/next')  #: for allowing client to fetch for the next batch
    f.expose_endpoint('/fit')  #: for signaling the backend to fit on the labeled data
    f.expose_endpoint('/feed')  #: for signaling the backend to fit on the labeled data

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
        f.block()
