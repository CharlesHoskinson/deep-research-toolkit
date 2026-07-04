"""Optional programmatic claim extraction via a Backend (only meaningful
under llm.provider=local). The verbatim gate is applied here too, so an
off-label local model can only under-produce, never corrupt the corpus."""
from __future__ import annotations

import json
from pathlib import Path

from ..compiler.dossier import source_text_for, verbatim_ok
from ..compiler.schema import normalize_evidence

_SYSTEM = (
    "You extract atomic, evidence-backed claims from research text. Rules: "
    "(1) one checkable assertion per claim; (2) every supporting_evidence quote MUST be "
    "copied verbatim (an exact substring) from the chunk text -- never paraphrase; "
    "(3) merge entity mentions that refer to the same thing; (4) do not force a claim "
    "the text does not support. Output ONLY a JSON array of claim objects."
)


def build_extraction_prompt(chunks: list[dict]) -> tuple[str, str]:
    schema = ('[{"claim_id": "c_0001", "claim": "...", "claim_type": "architectural|empirical|'
              'definitional|comparative", "confidence": "high|medium|low", "supporting_evidence": '
              '[{"locator": "<node_id>", "quote": "<verbatim substring>", "url": "<source_url or null>"}]}]')
    body = "\n\n".join(f"[{c.get('node_id') or c.get('locator')}]\n{c.get('text', '')}" for c in chunks)
    user = f"Output schema:\n{schema}\n\nChunks:\n{body}"
    return _SYSTEM, user


def parse_claims_response(text: str) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    return json.loads(text[start:end + 1])


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_claims_to_run(run_dir, producer: str, config, backend) -> dict:
    run_dir = Path(run_dir)
    source_id = run_dir.name if producer == "web" else \
        json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")).get("document_id", run_dir.name)
    chunks = _read_jsonl(run_dir / "chunks.jsonl")
    system, user = build_extraction_prompt(chunks)
    claims = parse_claims_response(backend.complete(system, user))

    kept, dropped = [], []
    for claim in claims:
        refs = normalize_evidence(claim, producer, source_id)
        ok = bool(refs) and all(
            verbatim_ok(ref.quote, source_text_for(
                {"producer": ref.producer, "source_id": ref.source_id, "page": ref.page}, config))
            for ref in refs
        )
        (kept if ok else dropped).append(claim)

    with open(run_dir / "claims.jsonl", "w", encoding="utf-8") as f:
        for claim in kept:
            claim.setdefault("schema_version", "1.0")
            f.write(json.dumps(claim, ensure_ascii=False) + "\n")
    return {"written": len(kept), "dropped": [c.get("claim_id") for c in dropped]}
