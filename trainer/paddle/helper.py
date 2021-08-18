import paddle


def create_dataloader(
    dataset: 'paddle.io.Dataset',
    mode: str = 'train',
    batch_size: int = 8,
    shuffle: bool = True,
    drop_last: bool = True,
    **kwargs,
):
    return paddle.io.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
