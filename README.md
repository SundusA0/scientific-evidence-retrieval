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

## Candidate-pool optimization

We evaluated reranking 10, 20, and 50 BM25 candidates using the same fine-tuned cross-encoder. Ranking quality was evaluated on 81 held-out training-side validation queries. Latency was measured separately on 50 SciFact training queries using Apple MPS.

| Candidates | Validation nDCG@10 | Validation Recall@10 | p50 latency | p95 latency |
|---|---:|---:|---:|---:|
| 50 | 0.7477 | 0.8210 | 155.71 ms | 175.72 ms |
| 20 | 0.7560 | 0.8395 | 66.05 ms | 74.47 ms |
| 10 | 0.7378 | 0.8025 | 33.91 ms | 36.52 ms |

The 20-candidate configuration reduced observed median latency by 57.6% relative to 50 candidates while producing the highest validation ranking metrics among the evaluated settings. It was selected as the interactive application default.

These are exploratory, single-run comparisons rather than statistically established ranking gains. Latency measurements were performed sequentially in separate runs. The original final test results use 50 candidates and are retained unchanged.

## Batch-size inference profiling

With candidate-k=20 fixed, batch sizes 8, 16, and 32 were benchmarked on Apple MPS.

| Batch size | p50 latency (50 queries) | p95 latency |
|---|---:|---:|
| 8 | 69.90 ms | 72.91 ms |
| 16 | 65.58 ms | 71.26 ms |
| 32 | 61.55 ms | 63.91 ms |

A second comparison using 100 training queries and 5 warmup queries produced:

| Batch size | p50 latency | p95 latency | Sequential throughput |
|---|---:|---:|---:|
| 16 | 65.73 ms | 69.92 ms | 15.09 queries/s |
| 32 | 62.05 ms | 64.36 ms | 16.14 queries/s |

Validation ranking metrics at batch size 32 remained nDCG@10=0.7560 and Recall@10=0.8395, matching the earlier batch-16 results to four decimal places.

Batch size 32 was selected for the local API. Measurements are sequential, hardware-specific, and not production concurrency benchmarks.

## Exploratory hybrid retrieval experiment

We investigated whether combining BM25 with pretrained dense retrieval
improves scientific evidence retrieval.

Using reciprocal rank fusion (RRF, constant 60), we combined the top 100
BM25 and dense results and selected 20 candidates for cross-encoder
reranking. Evaluation used the same 81 training-side validation queries.

| Metric | BM25 + reranker | Hybrid + reranker |
|---|---:|---:|
| Candidate Recall@20 | 0.8519 | 0.8951 |
| nDCG@10 | 0.7560 | 0.7655 |
| Recall@10 | 0.8395 | 0.8519 |

Paired exploratory uncertainty analysis:

| Metric | Mean difference | 95% bootstrap CI | Sign-flip p-value |
|---|---:|---:|---:|
| nDCG@10 | +0.0095 | [-0.0131, +0.0401] | 0.6782 |
| Recall@10 | +0.0123 | [-0.0247, +0.0494] | 1.0000 |

For nDCG@10, 2 queries improved, 73 were unchanged, and 6 worsened.

The largest improvement occurred for query 430, where dense retrieval
recovered a known relevant document ranked 71st by BM25.

The observed improvement was not statistically significant and depended
heavily on a small number of queries. Additional dense retrieval also
introduces unmeasured serving overhead.

Decision: retain BM25 plus the fine-tuned cross-encoder as the default.
Hybrid retrieval remains an exploratory research result rather than a
demonstrated improvement.

No further tuning was performed on the final test set.

## Public fine-tuned model

The trained model is publicly available on Hugging Face:

https://huggingface.co/Sundus08/scifact-minilm-cross-encoder

After installing project dependencies and downloading SciFact, run:

`SCI_EVIDENCE_RERANKER=Sundus08/scifact-minilm-cross-encoder SCI_EVIDENCE_DEVICE=cpu python -m uvicorn sci_evidence.api:app --host 127.0.0.1 --port 8000`

On Apple Silicon, use `SCI_EVIDENCE_DEVICE=mps`.

The model downloads automatically on first use. The local application reranks 20 BM25 candidates using batch size 32. Original held-out test results used 50 candidates.

Scores indicate predicted relevance, not scientific truth.

## BM25 corpus-scaling study: BEIR TREC-COVID

We measured how the project's **existing in-memory Python BM25 retriever** scales when indexing increasingly large subsets of the [BEIR TREC-COVID](https://github.com/beir-cellar/beir) scientific-document corpus. The full corpus contains 171,332 documents. This is a **retrieval-systems performance experiment**, not a new relevance-quality evaluation or a large-model training result.

### Measurements

Measured on an Apple Silicon MacBook Pro (16 GB RAM), running one local Python process per corpus size. Each experiment used the **same 50 shuffled TREC-COVID query texts** (seed 42), with five warm-up queries and BM25 top-10 retrieval. Corpus subsets comprise the **first N records** in the original corpus file; they are not random or representative samples. The index was rebuilt separately for every measurement.

| Indexed documents | Index build | Peak process RSS | Warm p50/query | Warm p95/query | Sequential queries/s |
|---:|---:|---:|---:|---:|---:|
| 5,000 | 0.31 s | 93.2 MiB | 2.19 ms | 3.20 ms | 455.2 |
| 20,000 | 0.68 s | 202.6 MiB | 6.57 ms | 9.86 ms | 152.0 |
| 50,000 | 1.79 s | 507.0 MiB | 19.19 ms | 27.38 ms | 52.7 |
| 171,332 (full) | 5.99 s | 1,612.8 MiB | 105.34 ms | 148.47 ms | 9.7 |

Moving from 5,000 to 171,332 documents increased corpus size **34.3x**, median retrieval latency **48.1x**, and peak process RSS **17.3x**. The full-corpus process reached about **1.58 GiB** peak RSS and **105 ms** median query latency, *before any neural reranking*. At the largest size, latency increased more than corpus size alone between the 50,000-document and full-corpus measurements, motivating further profiling rather than proving an asymptotic complexity class.

**Measurement boundaries:** Query timing includes in-process BM25 lookup only; it excludes file loading, index construction, HTTP overhead, model loading, and Transformer inference. Peak RSS reflects the *whole Python process* (including loaded documents), **not** the postings index by itself. Reported queries/second is derived from **sequential** query timings and does **not** measure concurrent server throughput. Only one 50-query sample per corpus size was taken, without confidence intervals; numbers are hardware- and workload-dependent. No relevance judgments were used, and no ranking accuracy claims follow from these results.

### Reproduce

1. Download `trec-covid.zip` from the [BEIR dataset downloads](https://github.com/beir-cellar/beir) and extract its `trec-covid/` directory under `data/` (the dataset is excluded from Git).
2. Install the project and test its scaling benchmark: `python -m pip install -e '.[dev]'` followed by `python -m pytest tests/test_corpus_scale.py`.
3. Execute the same four corpus-size runs:

```bash
for n in 5000 20000 50000 0; do
  python scripts/08_benchmark_corpus_scale.py --max-docs "$n" --queries 50 --warmup 5
done
```

The value `--max-docs 0` indexes all available documents. Machine-readable reports, including timestamps, environment details, corpus checksum, workload settings, and latency summaries, are saved as `results/trec_covid_bm25_scale_*.json`.

**Next profiling questions:** quantify per-stage costs inside BM25 (tokenization, postings traversal, scoring, top-k selection), explain latency growth on high-frequency query terms, and benchmark p95/p99 latency under concurrent HTTP requests separately. A distributed or production-scale serving claim would require additional experiments.

## API concurrency, inference profiling, and cross-request batching (exploratory)

We profiled the SciFact local FastAPI service with the publicly available [fine-tuned cross-encoder](https://huggingface.co/Sundus08/scifact-minilm-cross-encoder) running on Apple Silicon MPS. This section reports **single-machine measurements**, not production load-test guarantees or ranking-quality improvements. The API reranks the top 20 BM25 candidates per query.

### Concurrent HTTP requests

A controlled local server on `127.0.0.1:8001` served the same SciFact corpus (5,183 abstracts) and model. The benchmark used a closed-loop HTTP client with the indicated maximum outstanding requests. Each BM25 setting measured 200 requests; each reranking setting measured 50 requests, after warm-up. Figures are *client-observed HTTP* times, including server scheduling and response overhead.

| Method | Concurrent clients | Successful requests | Throughput (req/s) | HTTP p50 (ms) | HTTP p95 (ms) | HTTP p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 1 | 200/200 | 319.17 | 3.06 | 4.96 | 6.34 |
| BM25 | 4 | 200/200 | 343.05 | 11.36 | 16.04 | 23.95 |
| BM25 | 8 | 200/200 | 377.42 | 21.21 | 26.29 | 28.44 |
| BM25 | 16 | 200/200 | 354.25 | 45.70 | 62.95 | 72.52 |
| Reranker (MPS) | 1 | 50/50 | 15.58 | 63.20 | 70.54 | 71.61 |
| Reranker (MPS) | 4 | 50/50 | 16.41 | 239.94 | 277.91 | 287.71 |
| Reranker (MPS) | 8 | 50/50 | 16.83 | 471.72 | 506.00 | 524.33 |
| Reranker (MPS) | 16 | 50/50 | 15.94 | 977.76 | 1166.40 | 1167.66 |

Reranker throughput plateaued at roughly 16–17 requests/s in this exploratory run while response latency rose sharply with concurrency. Eight clients showed the highest **observed** throughput, but these short runs do not establish an optimal concurrency level. In particular, tail estimates from only 50 requests are unstable.

### Sequential inference-stage profiling

With 50 SciFact train-side queries, 5 warm-ups, and 20 BM25 candidates per query, the profiler synchronized MPS to attribute time to individual stages. This changes timing behavior relative to normal serving.

| Stage | Median (ms) | p95 (ms) |
|---|---:|---:|
| BM25 candidate retrieval | 2.33 | 4.11 |
| Tokenization | 3.48 | 4.21 |
| Input transfer to MPS | 1.24 | 1.39 |
| Transformer forward pass | 54.22 | 54.84 |
| Transfer scores to CPU | 0.23 | 0.29 |
| Sorting / ranking | <0.01 | 0.01 |
| Total | 61.50 | 63.08 |

The Transformer forward pass occupied about **88% of median total time** in this setup; it was the dominant measured component. Medians for separate stages need not sum exactly to the median total.

### Cross-request batching feasibility

We tested *offline simulated batching*: 80 fixed train-side queries, identical BM25 top-20 candidates, two repeats, and group sizes of 1, 2, 4, or 8 queries per model call. Candidate generation, HTTP request handling, real request arrivals, queue waiting, and model initialization were excluded. Each grouped forward pass scores up to `20 × group_size` query-document pairs.

| Queries per batch | Pairs per forward | Throughput (queries/s) | Amortized ms/query | Batch p95 (ms) | Top-5 agreement vs. unbatched |
|---|---:|---:|---:|---:|---:|
| 1 | 20 | 16.29 | 61.38 | 69.31 | 100% |
| 2 | 40 | 17.19 | 58.17 | 122.30 | 100% |
| 4 | 80 | 17.75 | 56.35 | 228.02 | 100% |
| 8 | 160 | 13.48 | 74.21 | 603.29 | 100% |

Four-query batches increased observed *offline* throughput by **~9%** compared with one-query batches, while eight-query batches were slower. Batch p95 is **not request-level p95**; a real microbatching queue would add wait time. Top-5 agreement on this specific workload does not guarantee identical logits or behavior on arbitrary inputs.

**Engineering decision:** Do not claim that microbatching improves online service latency or deploy a queue based on this pilot. First repeat comparisons across run orders and measure request-level queueing, overload behavior, throughput, and ranking equivalence under a real concurrent arrival workload. A bounded in-flight admission policy may be more valuable than increasing GPU batch size on this hardware.

### Reproduce

Start one Uvicorn server in a separate terminal with the documented public model, changing `--port` to `8001` if needed. Then run the benchmarks from the repository root with dependencies installed:

```bash
python scripts/09_benchmark_api_concurrency.py --url http://127.0.0.1:8001 --method bm25 --requests 200 --concurrency 1 4 8 16 --warmup 10
python scripts/09_benchmark_api_concurrency.py --url http://127.0.0.1:8001 --method rerank --requests 50 --concurrency 1 4 8 16 --warmup 5
# Stop the API process before standalone MPS profiling to avoid GPU interference.
python scripts/10_profile_reranker_stages.py --model Sundus08/scifact-minilm-cross-encoder --device mps --queries 50 --warmup 5 --candidate-k 20 --batch-size 32
python scripts/11_benchmark_cross_request_batching.py --model Sundus08/scifact-minilm-cross-encoder --device mps --queries 80 --candidate-k 20 --group-sizes 1 2 4 8 --warmup 3 --repeats 2
```

Raw reports are stored in `results/scifact_api_concurrency_*.json`, `results/scifact_reranker_stage_profile.json`, and `results/scifact_cross_request_batching_pilot.json`. Timings depend on hardware, system load, workload order, and short sample sizes; no distributed serving or production SLA claims are made.

## Reranker admission control and bounded waiting (exploratory)

The SciFact FastAPI service supports **opt-in, per-process reranker admission control**. By default `SCI_EVIDENCE_MAX_INFLIGHT_RERANK=0`, so admission limiting is disabled and existing behavior is preserved. With a positive value, concurrent reranking executions are limited; BM25 requests are not subject to that limit. If capacity is exhausted, a request receives HTTP **503** with `Retry-After: 1`, either immediately or after a configurable, bounded wait. This is **not** cross-request GPU microbatching or a distributed rate limiter.

Configuration:

| Environment variable | Default | Meaning |
|---|---:|---|
| `SCI_EVIDENCE_MAX_INFLIGHT_RERANK` | `0` | Maximum simultaneous reranking executions; `0` disables limiting |
| `SCI_EVIDENCE_RERANK_WAIT_MS` | `0` | Maximum time an admitted waiting request can wait for an inference slot (0–5000 ms) |
| `SCI_EVIDENCE_MAX_QUEUED_RERANK` | `0` | Maximum requests waiting for a slot (0–1000); does not include active inference |

The local benchmark ran one Uvicorn worker on Apple Silicon MPS, loading the published `Sundus08/scifact-minilm-cross-encoder` checkpoint with the 5,183-document SciFact corpus. Each concurrency setting used **200 requests** from the train-side query workload, after five warm-ups, without client retries. This is a **closed-loop fixed-count** load test on one computer, not an open-loop arrival model or a production service-level test.

### Observed acceptance and latency

The table contrasts (A) immediate rejection at capacity 1 with (B) one active inference, **four waiting positions**, and a **300 ms maximum wait**. The two policies were measured in separate runs. Successful responses and HTTP 503 rejections are both counted.

| Policy | Concurrent clients | HTTP 200 | HTTP 503 | Rejection rate | Accepted HTTP p50 |
|---|---:|---:|---:|---:|---:|
| Immediate rejection | 1 | 200 | 0 | 0% | 63.27 ms |
| Immediate rejection | 4 | 2 | 198 | 99% | 78.77 ms |
| Immediate rejection | 8 | 2 | 198 | 99% | 68.29 ms |
| Immediate rejection | 16 | 2 | 198 | 99% | 69.89 ms |
| Bounded wait | 1 | 200 | 0 | 0% | 63.16 ms |
| Bounded wait | 4 | 200 | 0 | 0% | 255.54 ms |
| Bounded wait | 8 | 6 | 194 | 97% | 259.17 ms |
| Bounded wait | 16 | 6 | 194 | 97% | 238.11 ms |

Under moderate concurrency (four clients), short bounded waiting raised completion from **2/200 to 200/200**, at the cost of queueing latency. At eight and sixteen clients, **97%** of requests were still rejected. Zero unexpected transport/API errors were recorded in these runs. The earlier *unrestricted* concurrency benchmark accepted all its requests, but used **50**, not 200, per setting; the measurements are not a controlled same-run throughput comparison.

**Interpretation and limitations:** Fast HTTP 503 responses can make request latency for the few accepted calls look deceptively good. At concurrency 4–16 the immediate-rejection run has only two successful observations per setting; the bounded-wait run has only six successful observations at concurrency 8 and 16. No trustworthy tail latency or steady-state acceptance estimate can be inferred from those small successful subsets. The fixed-count client dispatch also means a burst of immediate rejections shortens the run; observed rejection fractions are highly workload- and arrival-pattern-dependent. A `Retry-After` header is advisory, and the client in this experiment does not retry. These are measurements of overload **trade-offs**, not claims of improved throughput or production readiness.

### Reproduce

Install the project's API and ML dependencies and download SciFact as described above. For immediate rejection, start one server in a separate terminal:

```bash
SCI_EVIDENCE_RERANKER=Sundus08/scifact-minilm-cross-encoder \
SCI_EVIDENCE_DEVICE=mps \
SCI_EVIDENCE_MAX_INFLIGHT_RERANK=1 \
python -m uvicorn sci_evidence.api:app --host 127.0.0.1 --port 8002
```

In another terminal, run:

```bash
python scripts/12_benchmark_rerank_admission.py --url http://127.0.0.1:8002 --expected-capacity 1 --requests 200 --concurrency 1 4 8 16 --warmup 5 --output results/scifact_admission_capacity1.json
```

Stop the first Uvicorn server (Ctrl+C), then start the bounded-wait configuration:

```bash
SCI_EVIDENCE_RERANKER=Sundus08/scifact-minilm-cross-encoder \
SCI_EVIDENCE_DEVICE=mps \
SCI_EVIDENCE_MAX_INFLIGHT_RERANK=1 \
SCI_EVIDENCE_RERANK_WAIT_MS=300 \
SCI_EVIDENCE_MAX_QUEUED_RERANK=4 \
python -m uvicorn sci_evidence.api:app --host 127.0.0.1 --port 8003
```

In the client terminal:

```bash
python scripts/12_benchmark_rerank_admission.py --url http://127.0.0.1:8003 --expected-capacity 1 --requests 200 --concurrency 1 4 8 16 --warmup 5 --output results/scifact_admission_bounded_wait.json
```

Raw reports are committed in `results/scifact_admission_capacity1.json` and `results/scifact_admission_bounded_wait.json`. **Engineering decision:** leave limiting off by default. Admission control can bound in-flight GPU work, but acceptable queue length, wait time, rejection rate, retry behavior, and tail latency require service-specific load tests before deployment. Limits apply separately to each server process, not globally across multiple workers.
