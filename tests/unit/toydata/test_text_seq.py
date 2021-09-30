from finetuner.toydata import _build_vocab, _text_to_int_sequence

texts = ['hello', 'hello world', 'goodbye world!']


def test_build_vocab():
    assert _build_vocab(texts, 1) == {'hello': 2, 'world': 3, 'goodbye': 4}
    assert _build_vocab(texts, 2) == {'hello': 2, 'world': 3}


def test_text_to_int_sequence():
    vocab = _build_vocab(texts)
    assert _text_to_int_sequence(texts[0], vocab) == [2]
    assert _text_to_int_sequence(texts[1], vocab) == [2, 3]
    assert _text_to_int_sequence(texts[2], vocab) == [4, 3]

    vocab = _build_vocab(texts, min_freq=2)
    assert _text_to_int_sequence(texts[0], vocab) == [2]
    assert _text_to_int_sequence(texts[1], vocab) == [2, 3]
    assert _text_to_int_sequence(texts[2], vocab) == [1, 3]

    vocab = _build_vocab(texts, min_freq=3)
    assert _text_to_int_sequence(texts[0], vocab) == [1]
    assert _text_to_int_sequence(texts[1], vocab) == [1, 1]
    assert _text_to_int_sequence(texts[2], vocab) == [1, 1]


def test_text_to_int_sequence_max_len_longer():
    vocab = _build_vocab(texts)
    assert _text_to_int_sequence(texts[0], vocab, 3) == [0, 0, 2]
    assert _text_to_int_sequence(texts[1], vocab, 3) == [0, 2, 3]
    assert _text_to_int_sequence(texts[2], vocab, 3) == [0, 4, 3]

    vocab = _build_vocab(texts, min_freq=2)
    assert _text_to_int_sequence(texts[0], vocab, 3) == [0, 0, 2]
    assert _text_to_int_sequence(texts[1], vocab, 3) == [0, 2, 3]
    assert _text_to_int_sequence(texts[2], vocab, 3) == [0, 1, 3]


def test_text_to_int_sequence_max_len_shorter():
    vocab = _build_vocab(texts)
    assert _text_to_int_sequence(texts[0], vocab, 1) == [2]
    assert _text_to_int_sequence(texts[1], vocab, 1) == [3]
    assert _text_to_int_sequence(texts[2], vocab, 1) == [3]
