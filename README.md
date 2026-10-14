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

## Optional API

A simple BM25 endpoint is included (not the reranker):

```bash
python -m pip install -e '.[api]'
python -m uvicorn sci_evidence.api:app --reload
```

Navigate to `http://127.0.0.1:8000/docs`.

## Attribution

Dataset: BEIR SciFact. Starting pretrained reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`; pretrained dense baseline: `sentence-transformers/all-MiniLM-L6-v2`. All results are from the user's local experiments, not independently reproduced on a separate machine.
