"""pool_gold: union + dedup of gate-passing claims from multiple models into
one gold set (Task 12). Pure -- wraps selfconsistency.union_claims with
min_support=1, so any model's gate-passed claim is gold and duplicates
collapse by normalized claim key.

Also covers the Anomaly-1 fix: both pooled teacher models independently mint
bNN_c_NNNN-style claim_ids, so union_claims (which dedups by claim TEXT, not
id) can pool two different-content claims that happen to share a raw id --
namespace_claim_ids tags ids per source model at pool time so that can never
happen, and renamespace_duplicate_claim_ids migrates already-pooled rows
that predate the fix."""
from deep_research_toolkit.evalkit import (
    duplicate_claim_id_groups,
    namespace_claim_ids,
    pool_gold,
    renamespace_duplicate_claim_ids,
    slugify_model_id,
)
from deep_research_toolkit.llm.response import CLAIM_MARKER_RE


def _c(text, s, e, claim_id=None):
    claim = {"claim": text, "supporting_evidence": [{"locator": "d#c1", "start_char": s, "end_char": e}]}
    if claim_id is not None:
        claim["claim_id"] = claim_id
    return claim


def test_pool_dedups_across_models():
    e4b = [_c("Leaders rotate each epoch", 0, 10)]
    qwen = [_c("leaders  rotate each epoch", 0, 10), _c("Followers verify", 20, 30)]
    pooled = pool_gold([e4b, qwen])
    texts = sorted(c["claim"].lower().strip() for c in pooled)
    assert len(pooled) == 2  # the rotate claim collapses; verify is added
    assert texts == ["followers verify", "leaders rotate each epoch"]  # first-seen row (e4b's) wins


# ---------------------------------------------------------------------------
# slugify_model_id / namespace_claim_ids: Anomaly 1, pool-time fix
# ---------------------------------------------------------------------------

def test_slugify_model_id_collapses_colon_and_other_unsafe_chars():
    assert slugify_model_id("gemma4:e4b") == "gemma4-e4b"
    assert slugify_model_id("qwen3:30b-a3b-instruct-2507-q4_K_M") == \
        "qwen3-30b-a3b-instruct-2507-q4_K_M"


def test_slugify_model_id_output_matches_claim_marker_charset():
    # The whole point: a namespaced id must still round-trip through
    # [claim:<id>] markers, whose id group is [A-Za-z0-9_\-\.]+ (no colon).
    slug = slugify_model_id("qwen3:30b-a3b, weird/chars!")
    assert CLAIM_MARKER_RE.match(f"[claim:{slug}]")


def test_namespace_claim_ids_prefixes_with_model_slug_and_preserves_fields():
    claims = [{"claim_id": "b00_c_0001", "claim": "x", "supporting_evidence": []}]
    out = namespace_claim_ids(claims, "gemma4:e4b")
    assert out[0]["claim_id"] == "gemma4-e4b.b00_c_0001"
    assert out[0]["claim"] == "x"
    # input is not mutated
    assert claims[0]["claim_id"] == "b00_c_0001"


def test_namespace_claim_ids_output_ids_match_claim_marker_regex():
    claims = [{"claim_id": "b00_c_0001", "claim": "x", "supporting_evidence": []}]
    out = namespace_claim_ids(claims, "qwen3:30b-a3b-instruct-2507-q4_K_M")
    cid = out[0]["claim_id"]
    assert CLAIM_MARKER_RE.match(f"[claim:{cid}]").group(1) == cid


def test_namespace_claim_ids_at_pool_time_prevents_cross_model_id_collision():
    # Both models mint the SAME raw claim_id for DIFFERENT content -- exactly
    # the Anomaly-1 collision. Namespacing before pool_gold must leave both
    # rows in the pool (dedup is by claim key, not id) but with distinct ids.
    e4b = namespace_claim_ids([_c("Leaders rotate each epoch", 0, 10, claim_id="b00_c_0001")], "gemma4:e4b")
    qwen = namespace_claim_ids([_c("Followers verify blocks", 0, 10, claim_id="b00_c_0001")], "qwen3:30b-a3b")
    pooled = pool_gold([e4b, qwen])
    ids = [c["claim_id"] for c in pooled]
    assert len(ids) == len(set(ids)) == 2
    assert set(ids) == {"gemma4-e4b.b00_c_0001", "qwen3-30b-a3b.b00_c_0001"}


# ---------------------------------------------------------------------------
# duplicate_claim_id_groups / renamespace_duplicate_claim_ids: --renamespace
# migration of already-committed (pre-fix) pooled-gold rows
# ---------------------------------------------------------------------------

def test_duplicate_claim_id_groups_finds_only_repeated_ids():
    claims = [
        {"claim_id": "b00_c_0001", "claim": "a"},
        {"claim_id": "b00_c_0002", "claim": "b"},
        {"claim_id": "b00_c_0001", "claim": "c"},  # collides with row 0
    ]
    groups = duplicate_claim_id_groups(claims)
    assert groups == {"b00_c_0001": [0, 2]}


def test_renamespace_disambiguates_colliding_ids_by_occurrence_order():
    # Row order encodes provenance here (see the function's docstring): the
    # union's output is model A's surviving claims followed by model B's, so
    # the first file-order occurrence of a collision is always model[0] and
    # the second is always model[1].
    claims = [
        {"claim_id": "b00_c_0001", "claim": "from model A"},
        {"claim_id": "b00_c_0002", "claim": "unique, untouched"},
        {"claim_id": "b00_c_0001", "claim": "from model B"},
    ]
    out = renamespace_duplicate_claim_ids(claims, ["gemma4:e4b", "qwen3:30b-a3b"])
    ids = [c["claim_id"] for c in out]
    assert ids == ["gemma4-e4b.b00_c_0001", "b00_c_0002", "qwen3-30b-a3b.b00_c_0001"]
    # non-id fields untouched
    assert out[0]["claim"] == "from model A"
    assert out[2]["claim"] == "from model B"
    assert len(ids) == len(set(ids))


def test_renamespace_is_a_noop_on_already_unique_ids():
    claims = [{"claim_id": "a"}, {"claim_id": "b"}]
    assert renamespace_duplicate_claim_ids(claims, ["m1", "m2"]) == claims


def test_renamespace_is_idempotent_across_two_passes():
    claims = [
        {"claim_id": "b00_c_0001", "claim": "from model A"},
        {"claim_id": "b00_c_0001", "claim": "from model B"},
    ]
    once = renamespace_duplicate_claim_ids(claims, ["gemma4:e4b", "qwen3:30b-a3b"])
    twice = renamespace_duplicate_claim_ids(once, ["gemma4:e4b", "qwen3:30b-a3b"])
    assert once == twice
    assert duplicate_claim_id_groups(twice) == {}


def test_renamespace_falls_back_to_dup_tag_beyond_known_models():
    claims = [{"claim_id": "x"}, {"claim_id": "x"}, {"claim_id": "x"}]
    out = renamespace_duplicate_claim_ids(claims, ["only-one-model"])
    ids = [c["claim_id"] for c in out]
    assert ids[0] == "only-one-model.x"
    assert ids[1] == "dup1.x"
    assert ids[2] == "dup2.x"
    assert len(ids) == len(set(ids))
