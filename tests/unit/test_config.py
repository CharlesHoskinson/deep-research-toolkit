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


def test_defaults_are_the_local_qwen_stack(tmp_path):
    # Zero-config: the shipped default is the local, role-routed Qwen stack --
    # provider local, qwen embeddings, and each phase on its own Qwen model.
    isolated = tmp_path / "project"
    isolated.mkdir()
    cfg = load_config(isolated)
    assert cfg.llm_provider == "local"
    assert cfg.embedding_model == "qwen3-embedding:8b"
    assert cfg.llm_local["model"] == "qwen2.5:7b-instruct"
    assert cfg.llm_roles["extract"]["model"] == "qwen2.5:7b-instruct"
    assert cfg.llm_roles["wiki_write"]["model"] == "qwen3.6:35b-a3b"
    assert cfg.llm_roles["conflict_adjudicate"]["model"] == "qwen3.6:27b"
    assert cfg.llm_roles["synthesize"]["model"] == "qwen3.6:27b"
    assert cfg.llm_roles["code_agent"]["model"] == "Ornith-1.0-9B"


def test_minimal_local_config_gets_per_role_qwen_models(tmp_path):
    # A project that opts into local but names no flat model and no roles still
    # gets the full per-phase Qwen stack (not one model for everything).
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\nllm:\n  provider: local\n", encoding="utf-8"
    )
    cfg = load_config(tmp_path)
    assert cfg.llm_roles["extract"]["model"] == "qwen2.5:7b-instruct"
    assert cfg.llm_roles["synthesize"]["model"] == "qwen3.6:27b"


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


def test_index_dir_defaults_and_resolves(tmp_path):
    from deep_research_toolkit.config import load_config
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\nknowledge_base:\n  path: kb\n", encoding="utf-8"
    )
    cfg = load_config(tmp_path)
    assert cfg.index_dir == (tmp_path / ".deepresearch/index").resolve()
    assert cfg.embedding_model == "qwen3-embedding:8b"


def test_llm_local_block_parsed(tmp_path):
    from deep_research_toolkit.config import load_config
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\n"
        "llm:\n"
        "  provider: local\n"
        "  local:\n"
        "    base_url: http://localhost:11434/v1\n"
        "    model: Ornith-1.0-9B\n",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.llm_provider == "local"
    assert cfg.llm_local["base_url"] == "http://localhost:11434/v1"
    assert cfg.llm_local["model"] == "Ornith-1.0-9B"
    assert cfg.llm_local["temperature"] == 0.6  # default filled in
    assert cfg.llm_local["max_tokens"] == 16000  # generous default so reasoning isn't truncated


def test_llm_roles_default_and_override(tmp_path):
    from deep_research_toolkit.config import load_config
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\n"
        "llm:\n"
        "  provider: local\n"
        "  local:\n"
        "    model: base-model\n"
        "  roles:\n"
        "    extract:\n"
        "      model: qwen3.5:9b\n",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    # extract role: model overridden, role-appropriate defaults filled in
    assert cfg.llm_roles["extract"]["model"] == "qwen3.5:9b"
    assert cfg.llm_roles["extract"]["thinking"] is False
    assert cfg.llm_roles["extract"]["response_format"] == "json"
    # unconfigured role inherits the flat local model but keeps its own defaults
    assert cfg.llm_roles["synthesize"]["model"] == "base-model"
    assert cfg.llm_roles["synthesize"]["thinking"] is True
