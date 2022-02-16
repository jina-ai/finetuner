from typing import Dict, List, Mapping, Sequence, Union

from .helper import AnyTensor


def get_device_pytorch(device: str):
    """Get Pytorch compute device.
    :param device: device name.
    """

    import torch

    # translate our own alias into framework-compatible ones
    return torch.device(device)


def to_device_pytorch(
    inputs: Union[AnyTensor, Mapping[str, AnyTensor], Sequence[AnyTensor]], device
) -> Union[AnyTensor, Dict[str, AnyTensor], List[AnyTensor]]:
    """Maps various input types to device.
    :param inputs: Inputs to be placed onto device.
    :param device: The torch device to be placed on.
    :return: The inputs on the specified device.
    """

    import torch

    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    elif isinstance(inputs, Mapping):
        return {k: v.to(device) for k, v in inputs.items()}
    elif isinstance(inputs, Sequence):
        return [x.to(device) for x in inputs]


def get_device_paddle(device: str):
    """Get Paddle compute device.
    :param device: device name.
    """

    import paddle

    # translate our own alias into framework-compatible ones
    if device == 'cuda':
        return paddle.CUDAPlace(0)
    elif device == 'cpu':
        return paddle.CPUPlace()
    else:
        raise ValueError(
            f'Device {device} not recognized, only "cuda" and "cpu" are accepted'
        )


def to_device_paddle(
    inputs: Union[AnyTensor, Mapping[str, AnyTensor], Sequence[AnyTensor]], device
) -> Union[AnyTensor, Dict[str, AnyTensor], List[AnyTensor]]:
    """Maps various input types to device.
    :param inputs: Inputs to be placed onto device.
    :param device: The paddle device to be placed on.
    :return: The inputs on the specified device.
    """

    import paddle

    if isinstance(inputs, paddle.Tensor):
        return paddle.to_tensor(inputs, place=device)
    elif isinstance(inputs, Mapping):
        return {k: paddle.to_tensor(v, place=device) for k, v in inputs.items()}
    elif isinstance(inputs, Sequence):
        return [paddle.to_tensor(x, place=device) for x in inputs]


def get_device_keras(device: str):
    """Get tensorflow compute device.

    :param device: device name.
    """

    import tensorflow as tf

    # translate our own alias into framework-compatible ones
    if device == 'cuda':
        device = '/GPU:0'
    elif device == 'cpu':
        device = '/CPU:0'
    else:
        raise ValueError(
            f'Device {device} not recognized, only "cuda" and "cpu" are accepted'
        )
    return tf.device(device)
