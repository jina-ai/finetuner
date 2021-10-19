def get_dataset(module, arity: int):
    if arity == 2:

        return getattr(module, 'SiameseDataset')
    elif arity == 3:

        return getattr(module, 'TripletDataset')
    else:
        raise NotImplementedError
