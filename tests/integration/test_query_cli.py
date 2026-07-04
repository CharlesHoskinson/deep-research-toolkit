import json
import os
import subprocess
import sys
from pathlib import Path

from deep_research_toolkit.common.frontmatter import write_okf

REPO = Path(__file__).resolve().parents[2]


def _run(cwd, *args):
    env = {**os.environ, "DRT_FAKE_EMBEDDER": "1"}
    script = REPO / "skills" / "retrieval-planner" / "scripts" / "query.py"
    return subprocess.run([sys.executable, str(script), *args], cwd=cwd, env=env,
                          capture_output=True, text=True)


def test_search_wiki_after_compile(tmp_path):
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\nknowledge_base:\n  path: kb\n  index_dir: idx\n", encoding="utf-8")
    write_okf(tmp_path / "kb" / "hydra.md",
              {"type": "Concept", "title": "Hydra", "timestamp": "t"}, "Hydra settlement layer.\n")
    compile_script = REPO / "skills" / "knowledge-compiler" / "scripts" / "compile.py"
    subprocess.run([sys.executable, str(compile_script)], cwd=tmp_path,
                   env={**os.environ, "DRT_FAKE_EMBEDDER": "1"}, check=True, capture_output=True)

    result = _run(tmp_path, "search-wiki", "settlement")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert any(r["path"] == "hydra.md" for r in payload)
