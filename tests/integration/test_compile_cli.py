import os
import subprocess
import sys
from pathlib import Path

from deep_research_toolkit.common.frontmatter import write_okf

REPO = Path(__file__).resolve().parents[2]


def test_compile_script_builds_index(tmp_path):
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\nknowledge_base:\n  path: kb\n  index_dir: idx\n", encoding="utf-8")
    write_okf(tmp_path / "kb" / "a.md", {"type": "Concept", "title": "A", "timestamp": "t"}, "body\n")
    # Force the deterministic embedder so the test needs no model download.
    env = {"DRT_FAKE_EMBEDDER": "1"}
    full_env = {**os.environ, **env}
    script = REPO / "skills" / "knowledge-compiler" / "scripts" / "compile.py"
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=full_env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "idx" / "knowledge.duckdb").is_file()
