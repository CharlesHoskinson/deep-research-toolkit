from deep_research_toolkit.compiler.embed import FakeEmbedder


def test_fake_embedder_is_deterministic_and_fixed_dim():
    e = FakeEmbedder()
    a = e.embed(["hydra", "cardano"])
    b = e.embed(["hydra", "cardano"])
    assert a == b
    assert len(a) == 2 and all(len(v) == e.dim for v in a)


def test_fake_embedder_distinguishes_texts():
    e = FakeEmbedder()
    v1, v2 = e.embed(["alpha", "completely different text"])
    assert v1 != v2
