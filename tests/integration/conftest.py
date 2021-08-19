import pytest


@pytest.fixture
def params():
    return {
        'input_dim': 28,
        'output_dim': 32,
        'epochs': 5,
        'batch_size': 256,
        'feature_dim': 128,
        'learning_rate': 0.01,
    }
