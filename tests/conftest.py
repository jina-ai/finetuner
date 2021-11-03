import pytest
import tensorflow as tf


@pytest.fixture(autouse=True)
def clear_session():
    tf.keras.backend.clear_session()
    config_proto = tf.ConfigProto()
    config_proto.graph_options.rewrite_options.arithmetic_optimization = (
        tf.core.protobuf.rewriter_config_pb2.RewriterConfig.OFF
    )
    session = tf.Session(config=config_proto)
    tf.python.keras.backend.set_session(session)
