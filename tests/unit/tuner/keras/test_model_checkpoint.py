import copy
import os
import time

import keras
import pytest
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import CosineDecay

from finetuner.tuner.base import BaseTuner
from finetuner.tuner.callback import BestModelCheckpoint, TrainingCheckpoint
from finetuner.tuner.keras import KerasTuner
from finetuner.tuner.state import TunerState


@pytest.fixture(scope='module')
def keras_model() -> BaseTuner:
    embed_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=()),
            tf.keras.layers.Dense(10, activation='relu'),
        ]
    )
    return embed_model


def test_save_on_every_epoch_end(keras_model: BaseTuner, tmpdir):
    checkpoint = TrainingCheckpoint(save_dir=tmpdir)
    tuner = KerasTuner(embed_model=keras_model)
    tuner.state = TunerState(epoch=0, batch_index=2, current_loss=1.1)
    checkpoint.on_epoch_end(tuner)
    assert os.listdir(tmpdir) == ['saved_model_epoch_01']
    tuner.state = TunerState(epoch=1, batch_index=2, current_loss=0.5)
    checkpoint.on_epoch_end(tuner)
    assert os.listdir(tmpdir) == ['saved_model_epoch_02']


def test_same_model(keras_model: BaseTuner, tmpdir):

    tuner = KerasTuner(keras_model)
    checkpoint = TrainingCheckpoint(save_dir=tmpdir)
    tuner.state = TunerState(epoch=1, batch_index=2, current_loss=1.1)
    checkpoint.on_epoch_end(tuner)

    new_model = keras.models.load_model(os.path.join(tmpdir, 'saved_model_epoch_02'))

    for l1, l2 in zip(new_model.layers, keras_model.layers):
        assert l1.get_config() == l2.get_config()
        assert len(l1.weights) == len(l2.weights)
        for idx in range(len(l1.weights)):
            assert (l1.get_weights()[idx] == l2.get_weights()[idx]).all()


def test_load_model(keras_model: BaseTuner, tmpdir):
    def get_optimizer_and_scheduler(embdding_model):
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        scheduler = CosineDecay(initial_learning_rate=0.1, decay_steps=2)
        return (opt, scheduler)

    def get_optimizer_and_scheduler_different_parameters(embdding_model):
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        scheduler = CosineDecay(initial_learning_rate=0.01, decay_steps=3)
        return (opt, scheduler)

    new_model = copy.deepcopy(keras_model)

    before_stop_tuner = KerasTuner(
        keras_model, configure_optimizer=get_optimizer_and_scheduler
    )
    before_stop_tuner.state = TunerState(epoch=10, batch_index=2, current_loss=1.1)

    after_stop_tuner = KerasTuner(
        new_model, configure_optimizer=get_optimizer_and_scheduler_different_parameters
    )
    after_stop_tuner.state = TunerState(epoch=0, batch_index=2, current_loss=1.1)

    checkpoint = TrainingCheckpoint(save_dir=tmpdir)
    checkpoint.on_epoch_end(before_stop_tuner)

    checkpoint.load(after_stop_tuner, os.path.join(tmpdir, 'saved_model_epoch_11'))

    assert after_stop_tuner.state.epoch == 11

    for l1, l2 in zip(
        before_stop_tuner.embed_model.layers, after_stop_tuner.embed_model.layers
    ):
        assert l1.get_config() == l2.get_config()
        assert len(l1.weights) == len(l2.weights)
        for idx in range(len(l1.weights)):
            assert (l1.get_weights()[idx] == l2.get_weights()[idx]).all()


def test_save_best_only(keras_model: BaseTuner, tmpdir):

    checkpoint = BestModelCheckpoint(save_dir=tmpdir, monitor='current_loss')
    tuner = KerasTuner(embed_model=keras_model)
    tuner.state = TunerState(epoch=0, batch_index=2, current_loss=1.1)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert os.listdir(tmpdir) == ['best_model_current_loss']
    creation_time = os.path.getmtime(os.path.join(tmpdir, 'best_model_current_loss'))
    tuner.state = TunerState(epoch=1, batch_index=2, current_loss=1.5)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert creation_time == os.path.getmtime(
        os.path.join(tmpdir, 'best_model_current_loss')
    )
    tuner.state = TunerState(epoch=2, batch_index=2, current_loss=0.5)
    time.sleep(2)
    checkpoint.on_train_batch_end(tuner)
    checkpoint.on_epoch_end(tuner)
    assert creation_time < os.path.getmtime(
        os.path.join(tmpdir, 'best_model_current_loss')
    )


def test_load_best_model(keras_model: BaseTuner, tmpdir):

    new_model = copy.deepcopy(keras_model)
    checkpoint = BestModelCheckpoint(tmpdir)

    before_tuner = KerasTuner(embed_model=keras_model)
    before_tuner.state = TunerState(epoch=0, batch_index=2, current_loss=1.1)

    checkpoint.on_val_batch_end(before_tuner)
    checkpoint.on_epoch_end(before_tuner)

    after_tuner = KerasTuner(embed_model=new_model)
    after_tuner.state = TunerState(epoch=1, batch_index=2, current_loss=0)

    assert os.listdir(tmpdir) == ['best_model_val_loss']
    checkpoint.load_model(after_tuner, fp=os.path.join(tmpdir, 'best_model_val_loss'))

    for l1, l2 in zip(after_tuner.embed_model.layers, before_tuner.embed_model.layers):
        assert l1.get_config() == l2.get_config()
        assert len(l1.weights) == len(l2.weights)
        for idx in range(len(l1.weights)):
            assert (l1.get_weights()[idx] == l2.get_weights()[idx]).all()
