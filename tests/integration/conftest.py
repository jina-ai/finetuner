import pytest


@pytest.fixture
def params():
    return {
        'input_dim': 28,
        'output_dim': 8,
        'epochs': 2,
        'batch_size': 256,
        'feature_dim': 32,
        'learning_rate': 0.01,
        'num_train': 1000,
        'num_eval': 1000,
        'num_predict': 100,
        'max_seq_len': 10,
    }
