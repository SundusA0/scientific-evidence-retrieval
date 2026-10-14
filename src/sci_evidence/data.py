"""Load the BEIR SciFact corpus, queries, and relevance judgments."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
REQUIRED_FILES = (
    "corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv"
)


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return (self.title + " " + self.text).strip()


@dataclass
class SciFactDataset:
    documents: list[Document]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_scifact(root: Path | str, split: str = "test") -> SciFactDataset:
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    root = Path(root)
    if not (root / "corpus.jsonl").is_file():
        raise FileNotFoundError(f"Missing {root / 'corpus.jsonl'}. Run: evidence download")
    documents = [
        Document(id=str(r["_id"]), title=r.get("title", ""), text=r.get("text", ""))
        for r in _load_jsonl(root / "corpus.jsonl")
    ]
    all_queries = {str(r["_id"]): r["text"] for r in _load_jsonl(root / "queries.jsonl")}
    judgments: dict[str, dict[str, int]] = {}
    with (root / "qrels" / f"{split}.tsv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            score = int(row["score"])
            if score > 0:
                judgments.setdefault(qid, {})[did] = score

    # Evaluate only judged queries: BEIR convention. Do not claim to judge all papers.
    queries = {qid: all_queries[qid] for qid in judgments if qid in all_queries}
    return SciFactDataset(documents=documents, queries=queries,
                          qrels={qid: judgments[qid] for qid in queries})


def download_scifact(root: Path | str) -> Path:
    """Download selected known files only, without unsafe ZIP path extraction."""
    root = Path(root)
    if all((root / rel).is_file() for rel in REQUIRED_FILES):
        return root
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading BEIR SciFact from {SCIFACT_URL}")
    with tempfile.TemporaryDirectory() as temp:
        archive_file = Path(temp) / "scifact.zip"
        req = urllib.request.Request(SCIFACT_URL, headers={"User-Agent": "SciEvidenceResearch/0.1"})
        with urllib.request.urlopen(req, timeout=120) as response, archive_file.open("wb") as output:
            shutil.copyfileobj(response, output)
        with zipfile.ZipFile(archive_file) as archive:
            for rel in REQUIRED_FILES:
                source = f"scifact/{rel}"
                if source not in archive.namelist():
                    raise ValueError(f"Expected {source} was not found in the BEIR download")
                entry = archive.getinfo(source)
                if entry.file_size > 100_000_000:
                    raise ValueError(f"Unexpectedly large dataset file: {source}")
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Zip entry is read as a stream; no paths from ZIP are used for output.
                with archive.open(entry) as inp, dest.open("wb") as out:
                    shutil.copyfileobj(inp, out)
    print(f"Dataset ready in {root}")
    return root


def tiny_demo_dataset() -> SciFactDataset:
    """Synthetic data for smoke tests only; never describe as scientific results."""
    docs = [
        Document("d1", "Solar panels", "Solar photovoltaic panels convert sunlight into electrical energy."),
        Document("d2", "Coffee and sleep", "Caffeine in coffee can delay sleep onset."),
        Document("d3", "Ocean water", "Seawater salinity depends on dissolved salts."),
        Document("d4", "Vaccine studies", "Clinical vaccine trials evaluate safety and effectiveness."),
        Document("d5", "Battery storage", "Lithium batteries store electrical energy for later use."),
    ]
    queries = {"q1": "Do solar panels convert sunlight to electricity?",
               "q2": "Can caffeine delay sleep?"}
    return SciFactDataset(docs, queries, {"q1": {"d1": 1}, "q2": {"d2": 1}})
