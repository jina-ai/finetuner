import pytest

import finetuner as ft
from finetuner.tailor.pytorch.projection_head import ProjectionHead
from finetuner.tuner.augmentation import vision_preprocessor
from finetuner.tuner.pytorch.losses import SiameseLoss, TripletLoss


@pytest.fixture
def default_model():
    import torchvision.models as models

    return models.resnet50(pretrained=False)


@pytest.mark.parametrize(
    "n_cls,n_epochs,loss,distance",
    [
        (5, 2, TripletLoss, 'cosine'),
    ],
)
def test_self_supervised_learning(
    default_model, create_easy_data_instance, n_cls, n_epochs, loss, distance
):
    # Prepare model and data
    data, vecs = create_easy_data_instance(n_cls)

    projection_head = ProjectionHead(in_features=2048)
    print("start fit")
    ft.fit(
        model=default_model,
        train_data=data,
        epochs=n_epochs,
        batch_size=len(data),
        loss=loss(),
        num_items_per_class=2,
        learning_rate=1e-2,
        preprocess_fn=vision_preprocessor,
        to_embedding_model=True,
        layer_name='adaptiveavgpool2d_173',
        projection_head=projection_head,
        input_size=(3, 224, 224),
    )
