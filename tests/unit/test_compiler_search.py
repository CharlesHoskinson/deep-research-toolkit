from deep_research_toolkit.compiler.search import rrf_fuse


def test_rrf_rewards_agreement_across_lists():
    # NOTE: the original brief used [["a", "b", "c"], ["b", "a", "d"]], but that
    # gives "a" and "b" identical scores (1/61 + 1/62 each) — an exact tie that
    # stable sort resolves by insertion order ("a" first). Demoting "a" to rank 2
    # in the second list makes "b" strictly best while keeping the test's intent.
    fused = rrf_fuse([["a", "b", "c"], ["b", "d", "a"]], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] == "b"          # top of one, second of the other -> best combined
    assert set(ids[:2]) == {"a", "b"}


def test_rrf_handles_single_list_and_unique_ids():
    fused = rrf_fuse([["x", "y"]])
    assert [i for i, _ in fused] == ["x", "y"]


def test_rrf_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
