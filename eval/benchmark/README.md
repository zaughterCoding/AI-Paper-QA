# Paper QA evidence benchmark v1

The application at Git tag `v0.1.0` (commit
`5bcc07487d822df9edfe0f7f6b326af19acf1890`) is the baseline. This benchmark is a
separate, versioned evaluation artifact. Rerun that application on these questions;
do not compare its old 35-question document hit rate directly with these scores.

## Scope and frozen assets

120 English questions over 20 pinned ML papers: 90 answerable, 30 unanswerable.
There are 45 development and 75 test questions. All 20 papers have directly
authored questions, rather than 15 papers acting only as distractors.

| Primary category | Count | What it exercises |
|---|---:|---|
| Terminology | 20 | Names, abbreviations, architecture and training terminology |
| Numeric | 21 | Counts, percentages, units, relative improvements, experimental conditions |
| Semantic | 29 | Paraphrase, causal explanation, corrected premises, 10 paired rewrites |
| Cross-section | 10 | Two necessary evidence groups in separated passages of one paper |
| Cross-paper | 10 | Two papers, attribution, differing methods, units and baselines |
| Unanswerable | 30 | 10 near-domain, 5 partial comparisons, 5 missing details, 5 ambiguous, 3 out-of-domain, 2 adversarial requests |

Primary categories are exclusive; tags can overlap. Paired rewrites and composite
questions are grouped into connected `family` components using shared annotated
evidence before assigning splits. A composite and its constituent facts cannot
cross dev/test, which the loader and tests enforce. Both splits cover every primary
category. These are grouped question/fact splits over a shared retrieval corpus,
NOT an unseen-paper generalization experiment. Use dev for tuning, test for final
comparison, and report the category breakdown alongside the overall result.

The fixed artifacts are `v1/questions.jsonl` and `v1/manifest.json`. The manifest
records the question hash, paper IDs, arXiv versions, attribution/licenses and
normalized corpus hashes. The loader rejects changed questions or changed paper
text. Whitespace is normalized with Python `" ".join(text.split())`; case,
punctuation and Unicode are preserved. Character offsets are Python Unicode
string offsets into that normalized original text, NOT PDF pages or byte offsets.

The questions were AI-authored and checked against local extracted source text.
Equivalent supporting passages were added during pre-release source review.
This is not an independently human-adjudicated benchmark. Literal alignment is
machine-checked, but does not prove the reference answer is semantically correct
or the relevance judgments exhaustive. Unanswerable rationales are explicit;
keyword absence alone is NOT used to certify unanswerability.

This version evaluates the **indexed extracted text**, which omits tables and
figures. A fact present only in an omitted table is outside this corpus. Better
PDF parsing/table ingestion can be evaluated as an additional track, but must not
silently change the corpus for the v1 leaderboard. No finite 120-question set
covers every possible user query; this version does not measure multilingual QA,
multi-turn state, image/table understanding, or production load/concurrency.

## Keep the papers local

Only manifests, questions, short evidence excerpts, evaluation code and compact
results are tracked. Paper bodies, downloads and raw runs are ignored. Do not
use `git add -f` to add them. Compact baseline files contain source offsets, not
retrieved paper text.

```bash
python -m scripts.fetch_benchmark
python -m scripts.benchmark --split all
git ls-files eval/corpus
```

The last command must list only `eval/corpus/sources.json`. The fetcher uses the
pinned arXiv revision and validates the extracted text before saving. It will not
overwrite a mismatched existing paper. arXiv HTML rendering can change even for
a pinned paper revision: if its hash changes, restore a matching local archive,
rather than updating the hash or accepting a different corpus. Repository files
alone do not guarantee upstream HTML will remain byte-for-byte recoverable.

## Evidence labels and fair budgets

Each positive question has one or more `evidence_groups`:

- All groups are required (AND): two-paper comparisons cannot pass with one paper.
- Each group lists acceptable alternative source spans (OR).
- A span is covered only if the returned original-text intervals jointly cover
  its entire text. Adjacent/overlapping chunks can jointly supply that evidence.
- Repeated contexts consume the budget but do not add evidence credit.
- Multiple valid sources can be alternatives. Unlabeled equivalent evidence may
  still be missed; inspect false misses and version any adjudication changes.

The standard track uses the first **5 context units and at most 900 whitespace
words**. These are final contexts after fusion/reranking/expansion, not initial
candidates. The last included context is truncated to the remaining word budget.
For generation runs, the scorer rejects over-budget contexts: the adapter must
apply the cap before the answer is generated, not trim evidence after generation.
Full-paper parent contexts cannot evade this cap. The score reports the actual
mean context word count. This is a reproducible word budget, not model-token cost.
Report tokenizer-based prompt usage separately when evaluating generation.

Parent/child and multi-hop systems export the original spans that reach the final
answer context. Query rewriting and hybrid retrieval do not alter labels. Systems
using summaries export original supporting evidence in a separately identified
provenance track; original-evidence coverage is not proof that a summary preserved
that information. Do not compare summarized-context cost directly to verbatim cost.
For graph-only or external-web systems, retain an explicit source-evidence mapping
and declare changed scope. This benchmark does not accept fabricated source text.

## Run and compare

The saved `v0.1.0` runs used Top-5 and at most 900 context words. All 731 stored
chunk vectors were verified against the cached embedding model (maximum absolute
difference approximately `1.15e-7`). The app matches the original baseline commit.

| Split | Cases (positive / negative) | Evidence hit | Evidence recall | Complete evidence | MRR |
|---|---:|---:|---:|---:|---:|
| Development | 45 (34 / 11) | 64.71% | 60.29% | 55.88% | 0.4931 |
| Test | 75 (56 / 19) | 60.71% | 57.14% | 53.57% | 0.3128 |

Machine-readable runs: [development](v1/baselines/v0.1.0-dev.json) and
[test](v1/baselines/v0.1.0-test.json). These are retrieval-only runs. Answerability
and generation scores are unavailable until actual answers/decisions are exported
and reviewed. The scorer's synthetic tests verify all four confusion-matrix cells;
they are not presented as application measurements.

Use the project's Python environment and a database containing exactly the 20
pinned documents. `scripts/run_benchmark.py` is the adapter for the existing
retrieval service and original 180-word/30-word-overlap chunks. It reads the corpus
and database, verifies document hashes and chunk coverage, checks all stored
vectors against the loaded cached model, warms retrieval once and saves a run.
It never ingests, deletes or re-embeds database rows. Model loading is offline;
install/cache the embedding model beforehand. `--skip-vector-verification` exists
for development only; official baselines verify vectors.

```bash
python -m scripts.run_benchmark --split dev --output eval/runs/candidate-dev.json
python -m scripts.benchmark --split dev --predictions eval/runs/candidate-dev.json \
  --compare eval/benchmark/v1/baselines/v0.1.0-dev.json

python -m scripts.run_benchmark --split test --output eval/runs/candidate-test.json
python -m scripts.benchmark --split test --predictions eval/runs/candidate-test.json \
  --compare eval/benchmark/v1/baselines/v0.1.0-test.json \
  --output eval/runs/comparison-test.json
```

The comparison re-scores both prediction sets with the same scorer and budget.
It validates matching benchmark/corpus hashes and exact query coverage. It reports
metric deltas and question IDs improved/regressed. These are descriptive paired
differences, not statistical significance claims. Inspect individual errors:
the sample size does not justify treating a one-question gain as a robust win.

Future adapters can call `score()` directly or export the JSON contract below;
they need not import the app, database or embedding libraries into the scorer.
Use separate clearly labeled budget tracks for different `--k`/`--max-words`.
Freeze model/retrieval configurations on dev before the final test run. Do not
provide reference answers or evidence annotations to the system under evaluation.

## Prediction contract

A run JSON has `metadata` and `predictions`. Copy `benchmark_sha256` from
`manifest.questions_sha256` and `corpus_sha256` from the manifest. Include system
commit/config/model IDs, scorer version or hash, timestamp and latency scope in
metadata; the baseline runner records these plus model-weight/tokenizer hashes,
dependency versions, hardware/platform and vector verification.

```json
{
  "metadata": {
    "benchmark_sha256": "<manifest.questions_sha256>",
    "corpus_sha256": "<manifest.corpus_sha256>",
    "system_name": "hybrid-reranker-candidate"
  },
  "predictions": [
    {
      "id": "paperqa-001",
      "contexts": [{"source_id": "attention-is-all-you-need", "start": 100, "end": 250}],
      "answered": true,
      "answer": "Actual system answer, optionally retained in the ignored run file",
      "answer_correct": false,
      "citation_supported": 1,
      "citation_count": 2,
      "claims_supported": 1,
      "claim_count": 3,
      "latency_ms": 120.0
    }
  ]
}
```

This is a shape example; its offsets and incomplete list are not a valid run.
Provide exactly one prediction per question in the selected split. Unknown,
missing and duplicate IDs fail validation. A context can instead contain
`source_id` and verbatim `text`; ambiguous repeated passages require offsets.
If both text and offsets are supplied they must agree. Model scores are optional
and never used as answerability thresholds by the scorer.

`answered` is optional for a retrieval-only run. Supply a Boolean for **every**
question for answerability evaluation: true means the system supplies the requested
substantive answer; false means abstention or a clarification request. Partial
comparisons with an explicit statement that the missing part cannot be established
should be judged as abstentions to the complete request. A corrected false premise
with sufficient evidence is an answer, not an abstention. Record the classification
protocol in run metadata. Do not infer this field from a nonempty retrieval list.

## Metrics and TP / FP / TN / FN

For answerability, the positive class is **answerable from the fixed corpus** and
the predicted positive is **system answered**:

| Gold | System action | Cell |
|---|---|---|
| Answerable | Answered | TP |
| Unanswerable | Answered | FP |
| Unanswerable | Abstained / clarified | TN |
| Answerable | Abstained | FN |

An incorrect substantive answer to an answerable question is still an
answerability TP, but fails answer correctness. Do not call this matrix answer
accuracy. The scorer reports precision, recall, false-positive/negative rates
and decision accuracy, and tests explicitly exercise all four cells. A real
system may produce zero examples in a cell; the benchmark does not manufacture
errors to populate every cell.

Retrieval metrics use positives only:

- **Evidence hit rate:** at least one required evidence group was covered.
- **Evidence recall:** macro-average fraction of required groups covered.
- **Complete evidence rate:** all required groups were covered.
- **MRR:** reciprocal rank of the first prefix that fully covers any group.

These measure annotated evidence coverage, not semantic answer quality. NDCG is
not reported: the dataset does not supply exhaustive graded chunk judgments, and
chunks themselves can change between systems. Negatives receive no artificial
retrieval relevance score; high similarity alone is not an answerability FP.

Generation fields are optional reviewed judgments, not an automatic LLM judge.
`answer_correct` must reflect the actual answer, including all required parts and
units. Record false for an answerable question that the system declines. Citation
precision is supported citation associations / all cited associations; claim
support rate is supported factual claims / all factual claims, including uncited
claims in the denominator. Out-of-range counts fail. Record evaluator identity,
rubric and, if using a model judge, its exact model/prompt in metadata. These are
not inferred from retrieval matches; missing judgments yield null scores and
explicit coverage counts. Review all cases for a complete system-quality report.

Latency is a single warm, sequential run (nearest-rank P95). It is a local smoke
measurement, not a production load test. Compare on the same hardware with the
same scope; generation or retry latency must not be compared with retrieval-only
latency. Raw model outputs and detailed runs belong in ignored `eval/runs/`.

## Validation and changes

```bash
python -m pytest tests/test_benchmark.py -q
python -m pytest -q
```

The tests cover alternatives, cross-chunk coverage, duplicate contexts, budgets,
all confusion-matrix cells, unjudged answers, malformed runs, paired regression
reporting, pinned-download drift and Git exclusion. The legacy benchmark is
unchanged. Corrections after v1 release require a versioned dataset/metric change
and re-evaluation of both systems; preserve old questions and results. New product
capabilities can add extension suites while retaining v1 as the common core.
