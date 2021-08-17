from typing import List, Dict, Optional, Any

import paddle

class PaddleDataIO:
    def collate_fn(self,
                   batch: List[Any],
                   tensor_names: Optional[str] = None,
                   mode: str = 'train',
                   **kwargs) -> Dict['paddle.Tensor']:
        """A custom collate function for data loading that formats the batch as
        a tuple tensors."""
        assert isinstance(batch, list)

        _tensor_names = tensor_names if tensor_names else list(batch[0].keys())

        ret = {}
        for key in _tensor_names:
            data = [x[key] for x in batch]
            ret[key] = paddle.stack(data)

        return ret
