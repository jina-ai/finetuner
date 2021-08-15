from tests.data_generator import fashion_doc_generator


def test_doc_generator():
    for d in fashion_doc_generator():
        print(d)
        break