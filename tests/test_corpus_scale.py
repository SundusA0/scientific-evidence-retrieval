"""Small end-to-end smoke test for the TREC-COVID corpus-size benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "08_benchmark_corpus_scale.py"


def _write_tiny_beir(path: Path) -> None:
    (path / "qrels").mkdir(parents=True)
    docs = [
        {"_id": "d1", "title": "Caffeine", "text": "Coffee affects sleep."},
        {"_id": "d2", "title": "Vaccines", "text": "Vaccines are studied."},
        {"_id": "d3", "title": "Solar", "text": "Solar panels generate power."},
    ]
    queries = [
        {"_id": "q1", "text": "Does coffee affect sleep?"},
        {"_id": "q2", "text": "What about vaccines?"},
    ]
    (path / "corpus.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8"
    )
    (path / "queries.jsonl").write_text(
        "".join(json.dumps(q) + "\n" for q in queries), encoding="utf-8"
    )
    (path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n", encoding="utf-8"
    )


def test_corpus_scale_small_beir(tmp_path):
    data = tmp_path / "tiny-beir"
    _write_tiny_beir(data)
    output = tmp_path / "metrics.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--data", str(data),
         "--max-docs", "2", "--queries", "2", "--warmup", "1",
         "--top-k", "2", "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert "BM25 on 2 documents" in completed.stdout
    assert result["parameters"]["indexed_documents"] == 2
    assert result["parameters"]["available_documents"] == 3
    assert result["warm_sequential_query_latency"]["n_queries"] == 2
    assert result["index_build_seconds"] >= 0
    assert result["peak_process_rss_mib_after_index"] > 0
    assert len(result["corpus_sha256"]) == 64


def test_corpus_scale_rejects_bad_parameters(tmp_path):
    data = tmp_path / "tiny-beir"
    _write_tiny_beir(data)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--data", str(data),
         "--max-docs", "-2"], cwd=ROOT, capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "max-docs >= 0" in completed.stderr
