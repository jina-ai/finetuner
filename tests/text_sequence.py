from collections import defaultdict


def text_to_word_sequence(
    text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=' '
):
    text = text.lower()

    translate_dict = {c: split for c in filters}
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]


def build_vocab(texts, min_freq=1):
    all_tokens = defaultdict(int)
    for text in texts:
        seq = text_to_word_sequence(text)
        for s in seq:
            all_tokens[s] += 1

    # 0 for padding, 1 for unknown
    return {
        k: idx
        for idx, k in enumerate(
            (k for k, v in all_tokens.items() if v >= min_freq), start=2
        )
    }


def text_to_int_sequence(text, vocab, max_len=None):
    seq = text_to_word_sequence(text)
    vec = [vocab.get(s, 1) for s in seq]
    if max_len:
        if len(vec) < max_len:
            vec = [0] * (max_len - len(vec)) + vec
        elif len(vec) > max_len:
            vec = vec[-max_len:]
    return vec
