def get_framework(embed_model) -> str:
    if 'keras.' in embed_model.__module__:
        return 'keras'
    elif 'torch.' in embed_model.__module__:
        return 'torch'
    elif 'paddle.' in embed_model.__module__:
        return 'paddle'
    else:
        raise ValueError(
            f'can not determine the backend from embed_model from {embed_model.__module__}'
        )
