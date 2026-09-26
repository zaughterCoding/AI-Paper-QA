# Evaluation policy

- `v0.1.0` (`5bcc07487d822df9edfe0f7f6b326af19acf1890`) is the original application baseline.
- Use `eval/benchmark/v1` for every subsequent retrieval, chunking, embedding,
  reranking, query-transformation or answer-generation improvement. Read
  `eval/benchmark/README.md` before changing evaluation-related behavior.
- Tune on the development split. Run the frozen test split for final comparisons.
  Never silently edit questions, labels, splits, source texts or budgets to improve a score.
  If an annotation or corpus correction is necessary, release a new benchmark version
  and rerun both baseline and candidate; keep the original version and its results.
- Future systems export the documented prediction contract; annotations must not depend
  on database UUIDs, chunk boundaries, retrieval scores or a particular framework.
- Do not commit or upload paper bodies (including PDF, HTML or extracted text), model
  caches, raw retrieval dumps or secrets. `eval/corpus/` is ignored except for
  `sources.json`; `eval/runs/` is local-only. Check `git ls-files eval/corpus`
  before publishing. Short evidence annotations and source manifests may be tracked.
- Preserve the legacy `eval/questions.jsonl` and `scripts/evaluate.py` for historical
  reproduction; new experiments should use the evidence benchmark.
- Report measured results honestly: retrieval success, answerability decisions and
  answer correctness are distinct. Missing judgments are unavailable, never zero or perfect.
