# do not change this line manually
# this is managed by git tag and updated on every release
# NOTE: this represents the NEXT release version
__version__ = '0.0.3'

__default_tag_key__ = 'finetuner'

# define the high-level API: fit()
from typing import Optional, overload, TYPE_CHECKING


from .helper import AnyDNN, DocumentArrayLike
from .tuner.fit import TunerReturnType

# fit interface generated from Labeler + Tuner
# overload_inject_fit_tailor_tuner_start

# overload_inject_fit_tailor_tuner_end


# fit interface generated from Labeler + Tuner
# overload_inject_fit_labeler_tuner_start
@overload
def fit(
    embed_model: AnyDNN,
    train_data: DocumentArrayLike,
    clear_labels_on_start: bool = False,
    port_expose: Optional[int] = None,
    runtime_backend: str = 'thread',
    interactive: bool = True,
    head_layer: str = 'CosineLayer',
) -> None:
    ...


# overload_inject_fit_labeler_tuner_end


# fit interface generated from Tuner
# overload_inject_fit_tuner_start
@overload
def fit(
    embed_model: AnyDNN,
    train_data: DocumentArrayLike,
    eval_data: Optional[DocumentArrayLike] = None,
    epochs: int = 10,
    batch_size: int = 256,
    head_layer: str = 'CosineLayer',
) -> TunerReturnType:
    ...


# overload_inject_fit_tuner_end


def fit(*args, **kwargs) -> Optional[TunerReturnType]:
    if kwargs.get('interactive', False):
        kwargs.pop('interactive')
        from .labeler.fit import fit

        return fit(*args, **kwargs)
    else:
        from .tuner.fit import fit

        return fit(*args, **kwargs)
