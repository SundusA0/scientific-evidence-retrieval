import json
from pathlib import Path

from sci_evidence.data import load_scifact


def test_scifact_loads_only_judged_queries(tmp_path: Path):
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps({"_id": "doc10", "title": "Sample", "text": "Evidence."}) + "\n")
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"_id": "q10", "text": "Sample query"}) + "\n" +
        json.dumps({"_id": "unused", "text": "No judgments"}) + "\n")
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq10\tdoc10\t1\n")
    ds = load_scifact(tmp_path)
    assert len(ds.documents) == 1
    assert ds.documents[0].full_text == "Sample Evidence."
    assert ds.queries == {"q10": "Sample query"}
    assert ds.qrels == {"q10": {"doc10": 1}}
