# Phase 1: hybrid retrieval and cross-encoder reranking

The selected pipeline improves test complete-evidence coverage from 30/56 to
36/56 positives under the unchanged five-context, 900-word budget. It is slower
on CPU, regresses on terminology, and does not solve cross-paper completeness.
These are retrieval results, not measured answer accuracy or abstention quality.

## Protocol and implementation

- Dataset, evidence labels, corpus and scorer are unchanged from benchmark v1.
- Development: 45 questions, 34 positives. Test: 75 questions, 56 positives.
- Original chunks: 180 words, 30-word overlap, 731 chunks across 20 papers.
- Every run verifies all stored vectors against the cached embedding model;
  maximum absolute error is approximately `1.15e-7`.
- Dense retrieval is the original cosine search. Lexical retrieval uses PostgreSQL
  English full-text search over title/body, OR terms and `ts_rank_cd(..., 32)`;
  it is not BM25. No GIN index or additional search service was added.
- Each branch retrieves at most 30 candidates. RRF uses equal contributions
  `1 / (60 + rank)`, deduplicates chunk IDs and retains 30 fused candidates.
- Reranking uses `cross-encoder/ms-marco-MiniLM-L6-v2` revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`, on CPU, batch size 16,
  maximum pair length 512 tokens. Inputs are question and title/body; raw logits
  determine order. No score threshold or answerability classifier was fitted.
- Five development ablations were run at candidate depth 30. The selected
  `hybrid_rerank` configuration tied complete coverage and improved recall/MRR
  on dev. It was frozen before running the test comparison; no test-driven tuning.
- Test ran only the selected configuration and a same-machine dense control.
  Both dense controls reproduce the saved v0.1.0 retrieval metrics exactly.

The API defaults to the quality-oriented `hybrid_rerank` path. Set
`RETRIEVAL_MODE=dense` for substantially lower latency. The runner and historical
evaluation scripts retain explicit dense defaults for baseline reproduction.

## Development ablation

All final metrics below use Top-5 / 900 words. Candidate completeness uses up to
30 passages / 5400 words and is diagnostic only, not a comparable final score.

| Mode | Complete | Evidence recall | MRR | Candidate complete | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 55.88% | 60.29% | 0.4931 | 79.41% | 11.97 | 16.10 |
| FTS | 47.06% | 48.53% | 0.3711 | 79.41% | 257.23 | 503.76 |
| Hybrid | 55.88% | 60.29% | 0.4985 | 88.24% | 308.28 | 547.67 |
| Dense + rerank | 55.88% | 60.29% | 0.5147 | 79.41% | 795.62 | 979.95 |
| Hybrid + rerank | 55.88% | 63.24% | 0.5588 | 88.24% | 1112.61 | 1361.55 |

Fusion improved the candidate pool; reranking did not increase development
complete coverage. This is evidence of a remaining final-selection bottleneck,
not proof that a larger candidate pool alone will solve it. Candidate coverage
also does not prove all necessary evidence can fit the final context budget.

## Frozen test comparison

| Metric | v0.1.0 Dense | Hybrid + rerank | Difference |
|---|---:|---:|---:|
| Complete evidence | 53.57% (30/56) | 64.29% (36/56) | +10.71 pp |
| Evidence recall | 57.14% | 68.75% | +11.61 pp |
| Evidence hit | 60.71% | 73.21% | +12.50 pp |
| MRR | 0.3128 | 0.5563 | +0.2435 |
| Candidate complete, diagnostic | 83.93% | 91.07% | +7.14 pp |
| Mean final context words, all questions | 884.91 | 889.96 | +5.05 |
| Same-machine P50 retrieval ms | 11.77 | 997.08 | +985.31 |
| Same-machine P95 retrieval ms | 17.41 | 1207.82 | +1190.40 |

Category completeness (positives only):

| Category | Questions | Dense complete | Hybrid + rerank complete |
|---|---:|---:|---:|
| Terminology | 12 | 8/12 (66.67%) | 6/12 (50.00%) |
| Numeric | 12 | 8/12 (66.67%) | 10/12 (83.33%) |
| Semantic | 21 | 12/21 (57.14%) | 18/21 (85.71%) |
| Cross-section | 5 | 2/5 (40.00%) | 2/5 (40.00%) |
| Cross-paper | 6 | 0/6 | 0/6 |

Cross-paper evidence recall rises from 16.67% to 25.00%, but no question has all
required evidence. The 19 unanswerable test cases were retrieved and timed; without
actual answer/abstain decisions they have no retrieval relevance or confusion-matrix
score. Generation/citation correctness is likewise unmeasured.

Per-question evidence recall improves on 12 cases and regresses on 5. Regressions:
`paperqa-016`, `paperqa-024`, `paperqa-040`, `paperqa-043`, `paperqa-063`.
See [the paired comparison](comparison-test.json) for all IDs and metric deltas.
These are descriptive differences on a small, AI-authored, source-checked dataset;
no statistical significance or production-generalization claim is made.

## Reproduce

Prepare the pinned corpus/index and cached models as documented in the
[benchmark contract](../../../README.md) and project README. The runner uses a
read-only database transaction and does not ingest, delete or update corpus rows.

```bash
python -m scripts.benchmark --split all
python -m scripts.run_benchmark --split dev --retrieval-mode dense --candidate-k 30 --diagnostics --output eval/runs/dev-dense.json
# Repeat dev with fts, hybrid, dense_rerank and hybrid_rerank.
python -m scripts.run_benchmark --split test --retrieval-mode dense --candidate-k 30 --diagnostics --output eval/runs/test-dense.json
python -m scripts.run_benchmark --split test --retrieval-mode hybrid_rerank --candidate-k 30 --diagnostics --output eval/runs/test-hybrid_rerank.json
python -m scripts.benchmark --split test --predictions eval/runs/test-hybrid_rerank.json --compare eval/benchmark/v1/baselines/v0.1.0-test.json --output eval/runs/comparison-test.json
```

Runs are stored beside this report as `dev-<mode>.json` and `test-<mode>.json`.
They contain original-text offsets and scores, never paper bodies. Each run records
corpus/questions/scorer/pipeline/model hashes, dependencies and timing scope.
Development runs were measured on the uncommitted implementation based on
`0bb3e90`; their pipeline hashes identify that state. Final test runs use the clean
tracked implementation commit `a533c29809ac4157361467f0b515f64cb78958c9`.
Changes between dev and test were initialization locking, default wiring and
metadata/documentation; retrieval algorithms and selected parameters were unchanged.

Environment: Windows, Intel64 Family 6 Model 154 Stepping 3 CPU, Python 3.11.16,
PostgreSQL 16.15, torch 2.14.0+cpu, sentence-transformers 6.1.0, transformers
5.17.0, pgvector 0.5.0. Timing is one warm sequential pass per mode, including
query embedding, SQL, fusion and optional reranking. It excludes model loading,
vector validation, extra candidate diagnostics, scoring and LLM generation.
No concurrent tests were running during timing. These are local smoke timings,
not production concurrency measurements; wall-clock variation remains possible.

Validation: `python -m pytest -q` — **394 passed, 11 skipped**, with two upstream
deprecation warnings. Tests cover original dense behavior, real PostgreSQL FTS,
RRF deduplication, malformed reranker output, concurrent initialization and API
configuration/score provenance. Model quality is validated by the real-model
benchmark runs rather than the fake-model unit tests.
