"""This modules defines all kinds of exceptions raised in Finetuner."""


class DimensionMismatchException(Exception):
    """Dimensionality mismatch given input and output layers."""


class DeviceError(Exception):
    """Model is placed on the wrong device."""
