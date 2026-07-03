"""Regression tests for the two real bugs found during this suite's own
development: (1) re-running an earlier stage must not clobber a later
stage's entry, (2) two different callers writing into the SAME stage key
(e.g. knowledge-extraction's table/figure counts) must merge fields rather
than the second call wiping out the first's.
"""
from deep_research_toolkit.common.manifest import (
    load_manifest,
    start_manifest,
    update_stage,
    write_manifest,
)


def test_rerunning_a_stage_does_not_clobber_other_stages(tmp_path):
    write_manifest(tmp_path, start_manifest(tmp_path, "doc-1", "src.pdf", "sha256:abc"))
    update_stage(tmp_path, "stage-one", foo="bar")
    update_stage(tmp_path, "stage-two", baz=42)

    # simulate re-running stage-one from scratch
    write_manifest(tmp_path, start_manifest(tmp_path, "doc-1", "src.pdf", "sha256:abc"))
    update_stage(tmp_path, "stage-one", foo="bar-rerun")

    manifest = load_manifest(tmp_path)
    assert "stage-two" in manifest["stages"], "stage-two was clobbered by re-running stage-one"
    assert manifest["stages"]["stage-two"]["baz"] == 42
    assert manifest["stages"]["stage-one"]["foo"] == "bar-rerun"


def test_two_callers_writing_the_same_stage_merge_fields(tmp_path):
    write_manifest(tmp_path, start_manifest(tmp_path, "doc-1", "src.pdf", "sha256:abc"))
    update_stage(tmp_path, "knowledge-extraction", table_count=1)
    update_stage(tmp_path, "knowledge-extraction", figure_count=1)

    stage = load_manifest(tmp_path)["stages"]["knowledge-extraction"]
    assert stage["table_count"] == 1
    assert stage["figure_count"] == 1


def test_update_stage_creates_minimal_manifest_when_none_exists(tmp_path):
    update_stage(tmp_path, "some-stage", x=1)
    manifest = load_manifest(tmp_path)
    assert manifest is not None
    assert manifest["stages"]["some-stage"]["x"] == 1
    assert manifest["schema_version"]


def test_start_manifest_is_idempotent(tmp_path):
    write_manifest(tmp_path, start_manifest(tmp_path, "doc-1", "src.pdf", "sha256:abc"))
    update_stage(tmp_path, "stage-one", foo="bar")

    # calling start_manifest again must preserve stages already recorded
    m2 = start_manifest(tmp_path, "doc-1", "src.pdf", "sha256:abc")
    assert m2["stages"]["stage-one"]["foo"] == "bar"
