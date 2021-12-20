from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TunerState:
    """A data container representing containing information on the current run."""

    num_epochs: int = 0
    epoch: int = 0

    num_batches_train: int = 0
    num_batches_val: int = 0

    batch_index: int = 0
    current_loss: float = 0.0

    learning_rates: Dict[str, float] = field(default_factory=lambda: {})
