import abc
from typing import Any, Dict, List, Optional, Tuple, TypeVar

ModelStubType = TypeVar('ModelStubType', bound='_ModelStub')


class _ModelStub(metaclass=abc.ABCMeta):
    name: str
    description: str
    task: str
    output_dim: Optional[int]
    architecture: str
    options: Dict[str, Any]

    def __init__(self, **kwargs):
        self.options = kwargs

    @staticmethod
    def header() -> Tuple[str, ...]:
        """Get table header."""
        return 'name', 'task', 'output_dim', 'architecture', 'description'

    @classmethod
    def row(cls) -> Tuple[str, ...]:
        """Get table row."""
        return (
            cls.name,
            cls.task,
            str(cls.output_dim),
            cls.architecture,
            cls.description,
        )


class MLP(_ModelStub):
    """MLP model stub.

    :param input_size: Size of the input representations.
    :param hidden_sizes: A list of sizes of the hidden layers. The last hidden size is
        the output size.
    :param bias: Whether to add bias to each layer.
    :param activation: A string to configure activation function, `relu`, `tanh` or
        `sigmoid`. Set to `None` for no activation.
    :param l2: Apply L2 normalization at the output layer.
    """

    name = 'mlp'
    description = 'Simple MLP encoder trained from scratch'
    task = 'any'
    output_dim = '-'
    architecture = 'MLP'

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        bias: bool = True,
        activation: Optional[str] = None,
        l2: bool = False,
    ):
        super(MLP, self).__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            bias=bias,
            activation=activation,
            l2=l2,
        )


class ResNet50(_ModelStub):
    """ResNet50 model stub."""

    name = 'resnet50'
    description = 'Pretrained on ImageNet'
    task = 'image-to-image'
    output_dim = '2048'
    architecture = 'CNN'

    def __init__(self):
        super(ResNet50, self).__init__()


class ResNet152(_ModelStub):
    """ResNet152 model stub."""

    name = 'resnet152'
    description = 'Pretrained on ImageNet'
    task = 'image-to-image'
    output_dim = '2048'
    architecture = 'CNN'

    def __init__(self):
        super(ResNet152, self).__init__()


class EfficientNetB0(_ModelStub):
    """EfficientNetB0 model stub."""

    name = 'efficientnet_b0'
    description = 'Pretrained on ImageNet'
    task = 'image-to-image'
    output_dim = '1280'
    architecture = 'CNN'

    def __init__(self):
        super(EfficientNetB0, self).__init__()


class EfficientNetB4(_ModelStub):
    """EfficientNetB4 model stub."""

    name = 'efficientnet_b4'
    description = 'Pretrained on ImageNet'
    task = 'image-to-image'
    output_dim = '1280'
    architecture = 'CNN'

    def __init__(self):
        super(EfficientNetB4, self).__init__()


class OpenAICLIP(_ModelStub):
    """OpenAICLIP model stub."""

    name = 'openai/clip-vit-base-patch32'
    description = 'Pretrained on text image pairs by OpenAI'
    task = 'text-to-image'
    output_dim = '768'
    architecture = 'transformer'

    def __init__(self):
        super(OpenAICLIP, self).__init__()


class BERT(_ModelStub):
    """BERT model stub."""

    name = 'bert-base-cased'
    description = 'Pretrained on BookCorpus and English Wikipedia'
    task = 'text-to-text'
    output_dim = '768'
    architecture = 'transformer'

    def __init__(self):
        super(BERT, self).__init__()


class SentenceTransformer(_ModelStub):
    """SentenceTransformer model stub."""

    name = 'sentence-transformers/msmarco-distilbert-base-v3'
    description = 'Pretrained BERT, fine-tuned on MS Marco'
    task = 'text-to-text'
    output_dim = '768'
    architecture = 'transformer'

    def __init__(self):
        super(SentenceTransformer, self).__init__()
