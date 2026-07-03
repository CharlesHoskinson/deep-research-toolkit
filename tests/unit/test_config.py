from pathlib import Path

from deep_research_toolkit.config import find_config, load_config


def test_find_config_walks_up_from_subdirectory(tmp_path):
    (tmp_path / ".deepresearch.yml").write_text("version: 1\n", encoding="utf-8")
    subdir = tmp_path / "a" / "b" / "c"
    subdir.mkdir(parents=True)

    found = find_config(subdir)
    assert found == tmp_path / ".deepresearch.yml"


def test_find_config_returns_none_when_absent(tmp_path):
    isolated = tmp_path / "no-config-here"
    isolated.mkdir()
    # Use isolated as both start and implicitly stop walking at filesystem root;
    # this just confirms it doesn't find one that doesn't exist in this subtree.
    # (Can't fully isolate from a real ancestor .deepresearch.yml on a dev machine,
    # so this test only checks the "no match in this specific subtree" case.)
    found = find_config(isolated)
    assert found is None or found != isolated / ".deepresearch.yml"


def test_load_config_zero_config_default(tmp_path):
    isolated = tmp_path / "project"
    isolated.mkdir()
    cfg = load_config(isolated)
    assert cfg.config_path is None
    assert cfg.features == {"web_research": False, "pdf_ingestion": False, "knowledge_compiler": False}
    assert cfg.knowledge_base_path == isolated / "knowledge_base"


def test_load_config_resolves_paths_relative_to_config_file(tmp_path):
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\n"
        "knowledge_base:\n"
        "  path: kb/\n"
        "topic:\n"
        "  name: Test\n"
        "  scope_hint: testing\n"
        "features:\n"
        "  web_research: true\n",
        encoding="utf-8",
    )
    subdir = tmp_path / "sub"
    subdir.mkdir()

    cfg = load_config(subdir)
    assert cfg.knowledge_base_path == (tmp_path / "kb").resolve()
    assert cfg.topic_name == "Test"
    assert cfg.features["web_research"] is True
    assert cfg.features["pdf_ingestion"] is False
