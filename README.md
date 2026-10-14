# Scientific Evidence Retrieval and Evaluation

Reproducible information-retrieval experiment on **BEIR SciFact** (5,183 scientific abstracts and 300 held-out test queries). Includes BM25, pretrained dense retrieval, and a **fine-tuned cross-encoder reranker**, with train/validation/test separation. This is a *retrieval* system: it ranks candidate abstracts but does not verify whether a scientific claim is true.

## Results

The neural reranker fine-tuned on SciFact training queries outperformed BM25 on the held-out test split in the recorded single run. These are observed differences, not a demonstrated statistically significant gain; query-paired uncertainty analysis is available below.

| Method | Test nDCG@10 | Test Recall@10 |
|---|---:|---:|
| BM25 (first-stage retriever) | 0.6647 | 0.7849 |
| Pretrained `all-MiniLM-L6-v2` dense retriever | 0.6451 | 0.7833 |
| BM25 top-50 + fine-tuned `ms-marco-MiniLM-L-6-v2` cross-encoder | **0.6863** | **0.7903** |

Reranked minus BM25 deltas computed from unrounded values: **+0.0217 nDCG@10**, **+0.0053 Recall@10**. These methods differ: the dense model retrieves directly, whereas the reranker only reorders the top 50 BM25 candidates. The test candidate recall@50 is 0.8704, limiting achievable recall with this reranking architecture.

### Statistical uncertainty (300 paired test queries)

The reranker produced a higher point estimate than BM25 on both metrics, **but neither improvement is statistically significant at the 0.05 level**. Both paired 95% bootstrap confidence intervals include zero; exploratory two-sided sign-flip p-values exceed 0.05. These results do not establish a reliable improvement over BM25.

| Metric | Mean reranker − BM25 | Paired 95% bootstrap CI | Exploratory two-sided sign-flip p-value | Improved / tied / worsened queries |
|---|---:|---:|---:|---:|
| nDCG@10 | +0.0217 | [−0.0106, +0.0548] | 0.2052 | 67 / 182 / 51 |
| Recall@10 | +0.0053 | [−0.0327, +0.0440] | 0.7909 | 22 / 260 / 18 |

Confidence intervals reflect variation across the test queries for this **single trained checkpoint**; they do not include variability across retraining seeds. SciFact judgments are incomplete. These tests were exploratory and unadjusted for examining two metrics.

Held-out validation (81 queries) was used for model selection: BM25 nDCG@10 0.7211, pretrained cross-encoder 0.7038, 25-query smoke-trained cross-encoder 0.7146, and full training run 0.7477. The final test set was evaluated *after* choosing the full-data, one-epoch model. These recorded results should not be reused to tune further versions against the test set.

## Training design

- SciFact train split: 809 judged queries partitioned deterministically (seed 42) into 728 training and 81 validation queries.
- 835 labeled positive query-document training pairs and 2,505 top-50 BM25 mined **unjudged** pairs treated as weak negatives (3,340 total pairs). Unjudged does not mean verified irrelevant.
- Starting checkpoint: `cross-encoder/ms-marco-MiniLM-L-6-v2`. PyTorch AdamW, one epoch, batch size 8, initial learning rate 2e-5, maximum sequence length 256, binary cross-entropy with logits. Training executed on Apple Silicon MPS.
- SciFact test split: 300 queries, never used for gradient updates or choosing between the tested candidate models.

## Setup

Use native ARM64 Python 3.11 or 3.12 on Apple Silicon, or a compatible Python version on another platform. On Apple Silicon, check `python -c 'import platform; print(platform.machine())'` returns `arm64` if you intend to use MPS.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[ml,dev]'
python -m pytest
python -m sci_evidence download
```

To reproduce the retrieval baselines:

```bash
python -m sci_evidence benchmark --method bm25 --split test
python -m sci_evidence benchmark --method dense --split test --device cpu
```

For GPU acceleration on Apple Silicon, replace `--device cpu` with `--device mps` if `torch.backends.mps.is_available()` is true.

Train on the SciFact train-side partition (do not overwrite an existing output directory):

```bash
python scripts/03_train_reranker.py --epochs 1 --batch-size 8 --device mps --out runs/scifact_full_e1
python scripts/04_evaluate_reranker.py --model runs/scifact_full_e1 --split validation --device mps
```

**After the model and hyperparameters have been selected on validation**, run the final test evaluation once:

```bash
python scripts/04_evaluate_reranker.py --model runs/scifact_full_e1 --split test --device mps
python scripts/05_analyze_results.py
```

The final command computes paired bootstrap confidence intervals and exploratory two-sided sign-flip randomization p-values **from saved test query scores**. It does not download models or retrain. The statistical uncertainty from the completed run is documented in the Results section above. Do not infer statistical significance from a positive mean difference alone.

## Saved artifacts

- `results/scifact_test_bm25.json`: BM25 test metrics and per-query values
- `results/scifact_test_dense.json`: pretrained embedding baseline
- `results/scifact_validation_rerank_scifact_full_e1.json`: chosen model validation scores
- `results/scifact_test_rerank_scifact_full_e1.json`: chosen model final test scores, including paired query metrics
- `results/scifact_test_paired_statistics.json`: statistical analysis, once run
- `runs/scifact_full_e1/`: locally saved fine-tuned model (ignored by Git; can be recreated from training script)

## Limitations and responsible interpretation

Relevance judgments are incomplete; evaluation convention counts unjudged results as nonrelevant. Weak-negative training can introduce label noise. Reported test metrics are single-run observations, not replicated across different training seeds. The paired analysis assumes independent queries and uses exploratory, uncorrected tests for two metrics. No claims of truth verification, state-of-the-art performance, production readiness, or generalization beyond SciFact follow from these experiments.

## Interactive search interface and latency evaluation (Phase 4)

The interactive UI runs entirely on your **own laptop**, using the same SciFact corpus. It
can search with BM25 alone or optionally rerank the top 20 BM25 results (UI default) with your saved
fine-tuned cross-encoder. It does not use the SciFact test labels in the UI.

Install the API dependencies in the existing project virtual environment:

```bash
python -m pip install -e '.[dev,api]'
python -m pytest
python scripts/07_verify_results.py
```

The verification command checks the saved 300-query reports, per-query values, and
aggregate means/deltas. It **does not reproduce the training run**.

Launch the BM25 interface (requires downloaded `data/scifact`):

```bash
python -m uvicorn sci_evidence.api:app --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000/`. Stop with `Ctrl+C`. API documentation is at `/docs`.
To enable your **locally saved** trained model, use instead (in the terminal on
which the virtual environment is active):

```bash
SCI_EVIDENCE_RERANKER=runs/scifact_full_e1 SCI_EVIDENCE_DEVICE=mps \
  python -m uvicorn sci_evidence.api:app --host 127.0.0.1 --port 8000
```

The fine-tuned checkpoint is intentionally Git-ignored (not distributed with
this repository). On a machine without a trained checkpoint, the interface
supports BM25 only until the training script is run. Use
`SCI_EVIDENCE_DEVICE=cpu` if MPS is not available.

**Measure latency on training-side queries, not the final test set.** The default
experiment is warm, sequential query latency (model loading and index creation
are **excluded**). Query latency includes the first-stage BM25 retrieval and,
when enabled, the cross-encoder forward pass and score transfer to CPU.

```bash
python scripts/06_benchmark_latency.py --method bm25 --queries 10
python scripts/06_benchmark_latency.py --method rerank \
  --model runs/scifact_full_e1 --device mps --queries 10
```

For a more stable comparison use `--queries 100 --warmup 5` for **both** methods,
and save the resulting `results/scifact_latency_bm25.json` and
`results/scifact_latency_rerank.json` after measuring on your Mac. The reports
include the corpus SHA-256 checksum and the local model weights checksum when
available. They also specify machine architecture, Python version, dataset,
query count, model, and retrieval parameters. Latency is hardware-dependent;
the reported sequential queries/second is not concurrency throughput.

The browser shows server-side milliseconds for each search, rather than complete
user-perceived wall-clock latency. BM25's score and the reranker's logit are
not calibrated against each other. The API does not claim to determine which
papers support or refute a particular scientific claim.

## Future work

- Profile on MPS versus CPU and report warm-query p50 and p95 latency alongside ranking metrics.
- Investigate label noise introduced by BM25-mined unjudged training negatives.
- Run multiple training seeds **on training/validation only** and reserve an entirely new final evaluation dataset for any newly tuned model.
- Consider distillation or optimized serving if the cross-encoder overhead is too large.

## Attribution

Dataset: BEIR SciFact. Starting pretrained reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`; pretrained dense baseline: `sentence-transformers/all-MiniLM-L6-v2`. All results are from the user's local experiments, not independently reproduced on a separate machine.


## Inference latency on Apple Silicon

Measured on an Apple M4 MacBook using 50 sequential SciFact training queries per method, with 3 warmup queries. Model loading and initialization were excluded.

| Metric | BM25 (CPU) | Fine-tuned reranker (MPS) |
|---|---:|---:|
| p50 latency | 2.37 ms | 155.71 ms |
| p95 latency | 4.12 ms | 175.72 ms |
| Mean latency | 2.28 ms | 160.06 ms |
| Sequential throughput | 439.42 queries/s | 6.25 queries/s |

The reranker has approximately 66 times higher median latency. Its measured test nDCG@10 was 0.6863 versus 0.6647 for BM25, but the ranking improvement was not statistically significant.

These are machine-specific sequential measurements, not production serving benchmarks.
