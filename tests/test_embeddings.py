import embeddings


class FakeEmbedding:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, text: str):
        return FakeVector()


class FakeVector:
    def tolist(self):
        return [0.1, 0.2]


def test_question_model_is_loaded_lazily_and_cached(monkeypatch):
    created_models = []

    def factory(model_name):
        created_models.append(model_name)
        return FakeEmbedding(model_name)

    embeddings.get_question_embedding_model.cache_clear()
    monkeypatch.setattr(embeddings, "SentenceTransformer", factory)

    assert created_models == []
    assert embeddings.encode_question("question") == [0.1, 0.2]
    assert embeddings.encode_question("another question") == [0.1, 0.2]
    assert created_models == ["all-MiniLM-L6-v2"]

    embeddings.get_question_embedding_model.cache_clear()


def test_topic_model_is_loaded_only_when_requested(monkeypatch):
    created_models = []

    def factory(model_name):
        created_models.append(model_name)
        return FakeEmbedding(model_name)

    embeddings.get_topic_embedding_model.cache_clear()
    monkeypatch.setattr(embeddings, "SentenceTransformer", factory)

    assert created_models == []
    first_model = embeddings.get_topic_embedding_model()
    second_model = embeddings.get_topic_embedding_model()
    assert first_model is second_model
    assert created_models == ["all-mpnet-base-v2"]

    embeddings.get_topic_embedding_model.cache_clear()

