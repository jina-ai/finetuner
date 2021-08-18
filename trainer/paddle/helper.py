import paddle


def create_dataloader(
    dataset: 'paddle.io.Dataset',
    mode: str = 'train',
    batch_size: int = 8,
    shuffle: bool = True,
    drop_last: bool = True,
    **kwargs,
):
    sampler = paddle.io.BatchSampler(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )

    return paddle.io.DataLoader(dataset, batch_sampler=sampler)
